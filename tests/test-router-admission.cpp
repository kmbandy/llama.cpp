// Tests for the router's pure admission decision (tools/server/server-router-admission.cpp):
// spec §6 as a table (priority x blocker x placement shape) plus group footprints, host-RAM
// shortfalls, exclusive placement and pool ranking.

#include "server-router-admission.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static constexpr int64_t GB = 1024LL * 1024LL * 1024LL;

static const admission_priority PRIOS[] = { ADMISSION_PRIORITY_HIGHEST, ADMISSION_PRIORITY_MIDDLE, ADMISSION_PRIORITY_LOWEST };

static admission_slot slot(const std::string & id, const std::string & machine, int64_t free) {
    return { id, machine, "gpu:" + id, free };
}

static admission_resident resident(const std::string & name, const std::string & slot_id, int64_t vram, int64_t last_used) {
    admission_resident r;
    r.name      = name;
    r.vram      = { { slot_id, vram } };
    r.last_used = last_used;
    return r;
}

static admission_candidate pinned_on(const std::string & machine, const std::string & slot_id, int64_t vram) {
    admission_candidate c;
    c.machine = machine;
    c.vram    = { { slot_id, vram } };
    return c;
}

static std::string join(const std::vector<std::string> & v) {
    std::string out;
    for (const auto & s : v) {
        out += (out.empty() ? "" : ",") + s;
    }
    return out;
}

static std::string describe(const admission_result & r) {
    std::string out = std::string(admission_verdict_str(r.verdict)) + " slots=[" + join(r.slots) + "] victims=[" +
                      join(r.victims) + "] blocked=" + admission_block_str(r.blocked) + " by='" + r.blocked_by +
                      "' on='" + r.blocked_on + "' notify=[";
    for (const auto & n : r.notify) {
        out += n.claim_id + ":" + admission_notify_kind_str(n.kind) + " ";
    }
    return out + "] head=" + (r.queue_head ? "1" : "0");
}

static void check(bool ok, const std::string & what, const admission_result & r) {
    if (!ok) {
        fprintf(stderr, "FAIL %s\n  got: %s\n", what.c_str(), describe(r).c_str());
        abort();
    }
}

// --- the table ---------------------------------------------------------------------------------
//
// World: m1/g0 (40 GB free unless a resident fills it), m1/g1 (full, held by pinned "wall"),
// m2/g0 (20 GB free, the router itself holds a board claim on it). The load needs 10 GB with
// a 1 GB margin. The blocker under test always sits on m1/g0.

enum table_state { ST_FREE, ST_IDLE, ST_BUSY, ST_PINNED, ST_HELD, ST_CLAIM };
enum table_shape { SH_PINNED_ALIAS, SH_POOL, SH_POOL_OVERRIDE };

static const char * state_str(table_state s) {
    static const char * names[] = { "free", "idle", "busy", "pinned", "held", "foreign-claim" };
    return names[s];
}

static const char * shape_str(table_shape s) {
    static const char * names[] = { "pinned-alias", "pool", "pool+machine=m1" };
    return names[s];
}

static admission_input table_input(table_state st, table_shape sh, admission_priority prio) {
    admission_input in;
    in.alias        = "m";
    in.priority     = prio;
    in.margin_bytes = 1 * GB;
    in.slots        = { slot("m1/g0", "m1", 40 * GB), slot("m1/g1", "m1", 0), slot("m2/g0", "m2", 20 * GB) };
    in.machines     = { { "m1", "ram:m1", -1 }, { "m2", "ram:m2", -1 } };

    admission_resident wall = resident("wall", "m1/g1", 24 * GB, 1);
    wall.pinned             = true;
    in.residents.push_back(wall);
    // the router's own claim never blocks
    in.claims.push_back({ "c-router", "gpu:m2/g0", "llama-router", true, ADMISSION_PRIORITY_MIDDLE });

    if (st != ST_FREE && st != ST_CLAIM) {
        in.slots[0].free         = 8 * GB;
        admission_resident r     = resident(state_str(st), "m1/g0", 32 * GB, 5);
        r.busy                   = st == ST_BUSY;
        r.pinned                 = st == ST_PINNED;
        r.held                   = st == ST_HELD;
        in.residents.push_back(r);
    }
    if (st == ST_CLAIM) {
        in.claims.push_back({ "c1", "gpu:m1/g0", "session-a", false, ADMISSION_PRIORITY_MIDDLE });
    }

    if (sh == SH_PINNED_ALIAS) {
        in.candidates = { pinned_on("m1", "m1/g0", 10 * GB) };
    } else {
        in.candidates = admission_pool_candidates(in.slots, 10 * GB, 0);
        if (sh == SH_POOL_OVERRIDE) {
            in.machine = "m1";
        }
    }
    return in;
}

static void table_case(table_state st, table_shape sh, admission_priority prio) {
    const admission_result r = decide_admission(table_input(st, sh, prio));
    const std::string what = std::string(shape_str(sh)) + " / " + state_str(st) + " / " + admission_priority_str(prio);
    const bool highest = prio == ADMISSION_PRIORITY_HIGHEST;

    if (sh == SH_POOL) {
        // m1/g0 when it is free (most free VRAM); otherwise m2/g0, which needs no victim and
        // has no foreign claim; nobody is notified
        const std::string want = st == ST_FREE ? "m1/g0" : "m2/g0";
        check(r.verdict == ADMISSION_ADMIT && r.slots == std::vector<std::string>{ want } && r.victims.empty() &&
              r.notify.empty() && r.board_claims_to_take == std::vector<std::string>{ "gpu:" + want }, what, r);
        return;
    }

    // pinned alias, and a pool narrowed by the override to m1 (m1/g1 is walled off by a pin)
    switch (st) {
        case ST_FREE:
            check(r.verdict == ADMISSION_ADMIT && r.slots == std::vector<std::string>{ "m1/g0" } && r.victims.empty() &&
                  r.board_claims_to_take == std::vector<std::string>{ "gpu:m1/g0" }, what, r);
            break;
        case ST_IDLE: // idle residents are victims at every priority
            check(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.slots == std::vector<std::string>{ "m1/g0" } &&
                  r.victims == std::vector<std::string>{ "idle" }, what, r);
            break;
        case ST_BUSY: // busy only at highest
            if (highest) {
                check(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "busy" }, what, r);
            } else {
                check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_BUSY && r.blocked_by == "busy" &&
                      r.blocked_on == "m1/g0" && !r.blocked_on_ram && r.victims.empty() && r.notify.empty() &&
                      !r.queue_head, what, r);
            }
            break;
        case ST_PINNED: // never a victim, not even at highest
            check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_PINNED && r.blocked_by == "pinned" &&
                  r.victims.empty() && r.notify.empty() && r.queue_head == highest, what, r);
            break;
        case ST_HELD:
            check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_HELD && r.blocked_by == "held" &&
                  r.victims.empty() && r.notify.empty() && r.queue_head == highest, what, r);
            break;
        case ST_CLAIM: { // queue; yield at highest (head), wait at middle, silent at lowest
            check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_CLAIM && r.blocked_by == "session-a" &&
                  r.blocked_on == "gpu:m1/g0" && r.victims.empty() && r.board_claims_to_take.empty() &&
                  r.queue_head == highest, what, r);
            if (prio == ADMISSION_PRIORITY_LOWEST) {
                check(r.notify.empty(), what, r);
            } else {
                const admission_notify_kind want = highest ? ADMISSION_NOTIFY_YIELD : ADMISSION_NOTIFY_WAIT;
                check(r.notify.size() == 1 && r.notify[0].claim_id == "c1" && r.notify[0].holder == "session-a" &&
                      r.notify[0].kind == want, what, r);
            }
            break;
        }
    }
}

int main() {
    // priority strings, exactly
    assert(admission_priority_parse("highest") == ADMISSION_PRIORITY_HIGHEST);
    assert(admission_priority_parse("middle") == ADMISSION_PRIORITY_MIDDLE);
    assert(admission_priority_parse("lowest") == ADMISSION_PRIORITY_LOWEST);
    assert(!admission_priority_parse("high").has_value());
    assert(!admission_priority_parse("").has_value());
    assert(std::string(admission_priority_str(ADMISSION_PRIORITY_MIDDLE)) == "middle");
    assert(admission_input{}.priority == ADMISSION_PRIORITY_MIDDLE);

    // the table: priority x {free, idle, busy, pinned, held, foreign claim} x {pinned alias, pool, pool + override}
    for (int sh = SH_PINNED_ALIAS; sh <= SH_POOL_OVERRIDE; ++sh) {
        for (int st = ST_FREE; st <= ST_CLAIM; ++st) {
            for (admission_priority prio : PRIOS) {
                table_case((table_state) st, (table_shape) sh, prio);
            }
        }
    }

    // group footprint across two slots on two machines
    {
        admission_input in;
        in.alias    = "N";
        in.group    = "N";
        in.slots    = { slot("m1/g0", "m1", 4 * GB), slot("m2/g0", "m2", 2 * GB) };
        in.machines = { { "m1", "ram:m1", 10 * GB }, { "m2", "ram:m2", 1 * GB } };
        admission_candidate c;
        c.machine = "m1";
        c.vram    = { { "m1/g0", 12 * GB }, { "m2/g0", 10 * GB } };
        c.ram     = { { "m1", 6 * GB }, { "m2", 6 * GB } };
        in.candidates = { c };

        // resident group G: spine on m1/g0, worker on m2/g0, each with host RAM on its machine
        admission_resident spine = resident("G", "m1/g0", 20 * GB, 5);
        spine.group  = "G";
        spine.ram    = { { "m1", 4 * GB } };
        admission_resident worker = resident("G-w1", "m2/g0", 16 * GB, 2);
        worker.group = "G";
        worker.ram   = { { "m2", 8 * GB } };
        // older but only on m2/g0: evicting G already frees m2/g0, so it stays
        admission_resident other = resident("other", "m2/g0", 4 * GB, 1);
        other.ram    = { { "m2", 1 * GB } };
        // the load's own group member is never a victim, however old
        admission_resident own = resident("N-w0", "m1/g0", 30 * GB, 0);
        own.group    = "N";
        in.residents = { spine, worker, other, own };

        admission_result r = decide_admission(in);
        check(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "G" } &&
              r.slots == std::vector<std::string>{ "m1/g0", "m2/g0" } &&
              r.board_claims_to_take == std::vector<std::string>{ "gpu:m1/g0", "gpu:m2/g0", "ram:m1", "ram:m2" },
              "group footprint: one group victim frees both slots and both machines' RAM", r);

        // a busy worker makes the whole group busy: queue at middle naming the group ...
        in.residents[1].busy = true;
        r = decide_admission(in);
        check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_BUSY && r.blocked_by == "G",
              "group footprint: busy worker blocks at middle", r);
        // ... and the group goes at highest
        in.priority = ADMISSION_PRIORITY_HIGHEST;
        r = decide_admission(in);
        check(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "G" },
              "group footprint: busy group is a victim at highest", r);
        // a pinned member pins the group
        in.residents[1].busy   = false;
        in.residents[1].pinned = true;
        r = decide_admission(in);
        check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_PINNED && r.blocked_by == "G",
              "group footprint: pinned worker pins its group", r);

        // foreign claims on both of the group's GPUs: queue, one yield per claim
        in.residents[1].pinned = false;
        in.claims = { { "c1", "gpu:m1/g0", "s1", false, ADMISSION_PRIORITY_LOWEST },
                      { "c2", "gpu:m2/g0", "s2", false, ADMISSION_PRIORITY_HIGHEST } };
        r = decide_admission(in);
        check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_CLAIM && r.notify.size() == 2 &&
              r.notify[0].claim_id == "c1" && r.notify[1].claim_id == "c2" &&
              r.notify[0].kind == ADMISSION_NOTIFY_YIELD && r.notify[1].kind == ADMISSION_NOTIFY_YIELD,
              "group footprint: claims on both GPUs", r);
    }

    // a worker on the needed slot is evicted as its group (named after the spine)
    {
        admission_input in;
        in.alias      = "m";
        in.slots      = { slot("g0", "", 0), slot("g1", "", 0) };
        admission_resident w = resident("S-w", "g0", 12 * GB, 3);
        w.group       = "S";
        admission_resident s = resident("S", "g1", 12 * GB, 3);
        s.group       = "S";
        in.residents  = { w, s };
        in.candidates = { pinned_on("", "g0", 10 * GB) };
        const admission_result r = decide_admission(in);
        check(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "S" },
              "worker on slot -> group victim", r);
    }

    // RAM-only shortfall: VRAM fits, host RAM does not
    {
        admission_input in;
        in.alias    = "m";
        in.slots    = { slot("m1/g0", "m1", 40 * GB), slot("m1/g1", "m1", 40 * GB) };
        in.machines = { { "m1", "ram:m1", 2 * GB } };
        admission_candidate c = pinned_on("m1", "m1/g0", 10 * GB);
        c.ram = { { "m1", 8 * GB } };
        in.candidates = { c };
        admission_resident old_r = resident("r-old", "m1/g1", 8 * GB, 1);
        old_r.ram = { { "m1", 4 * GB } };
        admission_resident new_r = resident("r-new", "m1/g1", 8 * GB, 9);
        new_r.ram = { { "m1", 4 * GB } };
        admission_resident pin_r = resident("r-pin", "m1/g1", 1 * GB, 0);
        pin_r.ram    = { { "m1", 20 * GB } };
        pin_r.pinned = true;
        in.residents = { new_r, pin_r, old_r };

        admission_result r = decide_admission(in);
        check(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "r-old", "r-new" } &&
              r.slots == std::vector<std::string>{ "m1/g0" } &&
              r.board_claims_to_take == std::vector<std::string>{ "gpu:m1/g0", "ram:m1" },
              "ram shortfall: idle LRU victims", r);

        // more than every evictable resident frees: the pinned one is in the way
        in.candidates[0].ram = { { "m1", 30 * GB } };
        r = decide_admission(in);
        check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_PINNED && r.blocked_by == "r-pin" &&
              r.blocked_on == "m1" && r.blocked_on_ram, "ram shortfall: pinned in the way", r);

        // nobody to evict at all: plain capacity
        in.residents.clear();
        r = decide_admission(in);
        check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_CAPACITY && r.blocked_by.empty() &&
              r.blocked_on == "m1" && r.blocked_on_ram, "ram shortfall: capacity", r);

        // unknown free RAM never gates
        in.machines[0].free_ram = -1;
        r = decide_admission(in);
        check(r.verdict == ADMISSION_ADMIT, "ram unknown: admit", r);

        // a busy resident holding the RAM: queue at middle, victim at highest
        in.machines[0].free_ram = 2 * GB;
        in.candidates[0].ram    = { { "m1", 8 * GB } };
        admission_resident busy_r = resident("r-busy", "m1/g1", 1 * GB, 4);
        busy_r.ram  = { { "m1", 10 * GB } };
        busy_r.busy = true;
        in.residents = { busy_r };
        r = decide_admission(in);
        check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_BUSY && r.blocked_on_ram,
              "ram shortfall: busy at middle", r);
        in.priority = ADMISSION_PRIORITY_HIGHEST;
        r = decide_admission(in);
        check(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "r-busy" },
              "ram shortfall: busy at highest", r);

        // a foreign claim on the machine's RAM blocks a load that needs RAM ...
        in.priority  = ADMISSION_PRIORITY_MIDDLE;
        in.residents.clear();
        in.machines[0].free_ram = 40 * GB;
        in.claims = { { "cr", "ram:m1", "s-ram", false, ADMISSION_PRIORITY_MIDDLE } };
        r = decide_admission(in);
        check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_CLAIM && r.blocked_on == "ram:m1" &&
              r.notify.size() == 1 && r.notify[0].kind == ADMISSION_NOTIFY_WAIT, "ram claim blocks", r);
        // ... but not one that needs none
        in.candidates[0].ram.clear();
        r = decide_admission(in);
        check(r.verdict == ADMISSION_ADMIT, "ram claim: no ram need", r);
    }

    // a VRAM victim's host RAM counts toward the RAM shortfall: no extra RAM victim
    {
        admission_input in;
        in.alias    = "m";
        in.slots    = { slot("g0", "", 0), slot("g1", "", 40 * GB) };
        in.machines = { { "", "ram", 0 } };
        admission_candidate c = pinned_on("", "g0", 10 * GB);
        c.ram = { { "", 6 * GB } };
        in.candidates = { c };
        admission_resident v = resident("v", "g0", 12 * GB, 8);
        v.ram = { { "", 8 * GB } };
        admission_resident bystander = resident("bystander", "g1", 1 * GB, 1); // older, RAM only
        bystander.ram = { { "", 8 * GB } };
        in.residents = { v, bystander };
        const admission_result r = decide_admission(in);
        check(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "v" },
              "vram victim credited against ram", r);
    }

    // exclusive: every resident on the card goes, even with room to spare; VRAM is not gated
    {
        admission_input in;
        in.alias      = "x";
        in.exclusive  = true;
        in.slots      = { slot("g0", "", 40 * GB) };
        in.residents  = { resident("b", "g0", 1 * GB, 7), resident("a", "g0", 1 * GB, 2) };
        in.candidates = { pinned_on("", "g0", 10 * GB) };
        admission_result r = decide_admission(in);
        check(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "a", "b" },
              "exclusive: all overlapping, LRU order", r);

        admission_resident c = resident("c", "g0", 1 * GB, 1);
        c.busy = true;
        in.residents.push_back(c);
        r = decide_admission(in);
        check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_BUSY && r.blocked_by == "c",
              "exclusive: busy blocks at middle", r);
        in.priority = ADMISSION_PRIORITY_HIGHEST;
        r = decide_admission(in);
        check(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "a", "b", "c" },
              "exclusive: idle first, then busy at highest", r);

        in.residents.clear();
        in.slots[0].free           = 0;
        in.candidates[0].vram[0].bytes = 100 * GB;
        r = decide_admission(in);
        check(r.verdict == ADMISSION_ADMIT, "exclusive: VRAM not gated", r);
    }

    // an exclusive resident on the slot must go even when there is room
    {
        admission_input in;
        in.alias      = "m";
        in.slots      = { slot("g0", "", 40 * GB) };
        admission_resident x = resident("x", "g0", 1 * GB, 3);
        x.exclusive   = true;
        in.residents  = { x, resident("y", "g0", 1 * GB, 1) };
        in.candidates = { pinned_on("", "g0", 10 * GB) };
        const admission_result r = decide_admission(in);
        check(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "x" },
              "exclusive resident is a mandatory victim", r);
    }

    // highest on one slot: idle victims first, busy only if still short
    {
        admission_input in;
        in.alias    = "m";
        in.priority = ADMISSION_PRIORITY_HIGHEST;
        in.slots    = { slot("g0", "", 0) };
        admission_resident b = resident("busy-old", "g0", 10 * GB, 1);
        b.busy = true;
        in.residents  = { b, resident("idle-new", "g0", 5 * GB, 9) };
        in.candidates = { pinned_on("", "g0", 11 * GB) };
        admission_result r = decide_admission(in);
        check(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "idle-new", "busy-old" },
              "highest: idle then busy", r);
        in.candidates[0].vram[0].bytes = 4 * GB;
        r = decide_admission(in);
        check(r.victims == std::vector<std::string>{ "idle-new" }, "highest: busy spared when idle suffices", r);
    }

    // pool ranking
    {
        admission_input in;
        in.alias    = "m";
        in.priority = ADMISSION_PRIORITY_HIGHEST;
        // A: two idle victims; B: one busy victim -> A (busy victims cost more)
        in.slots = { slot("A", "", 0), slot("B", "", 0) };
        admission_resident b1 = resident("b1", "B", 20 * GB, 1);
        b1.busy = true;
        in.residents  = { resident("a1", "A", 6 * GB, 1), resident("a2", "A", 6 * GB, 2), b1 };
        in.candidates = admission_pool_candidates(in.slots, 11 * GB, 0);
        admission_result r = decide_admission(in);
        check(r.slots == std::vector<std::string>{ "A" } && r.victims.size() == 2, "pool: idle over busy", r);

        // C: one victim used at 9; D: one victim used at 3 -> D (staler victim)
        in.slots      = { slot("C", "", 0), slot("D", "", 0) };
        in.residents  = { resident("c1", "C", 20 * GB, 9), resident("d1", "D", 20 * GB, 3) };
        in.candidates = admission_pool_candidates(in.slots, 11 * GB, 0);
        r = decide_admission(in);
        check(r.slots == std::vector<std::string>{ "D" } && r.victims == std::vector<std::string>{ "d1" },
              "pool: least recently used victim", r);

        // C: one victim; D: none -> D, fewest victims beats free VRAM
        in.slots      = { slot("C", "", 5 * GB), slot("D", "", 12 * GB) };
        in.residents  = { resident("c1", "C", 20 * GB, 1) };
        in.candidates = admission_pool_candidates(in.slots, 11 * GB, 0);
        r = decide_admission(in);
        check(r.verdict == ADMISSION_ADMIT && r.slots == std::vector<std::string>{ "D" }, "pool: fewest victims", r);

        // no victims anywhere: most free VRAM, ties in input order
        in.residents.clear();
        in.slots      = { slot("E", "", 20 * GB), slot("F", "", 30 * GB), slot("G", "", 30 * GB) };
        in.candidates = admission_pool_candidates(in.slots, 11 * GB, 0);
        r = decide_admission(in);
        check(r.slots == std::vector<std::string>{ "F" }, "pool: most free VRAM, then input order", r);

        // a claimed slot is skipped while another fits
        in.claims = { { "cf", "gpu:F", "s", false, ADMISSION_PRIORITY_MIDDLE } };
        r = decide_admission(in);
        check(r.verdict == ADMISSION_ADMIT && r.slots == std::vector<std::string>{ "G" } && r.notify.empty(),
              "pool: claimed slot skipped", r);
    }

    // machine override that excludes the pinned alias's machine: nothing to place on
    {
        admission_input in;
        in.alias      = "m";
        in.machine    = "m2";
        in.slots      = { slot("m1/g0", "m1", 40 * GB) };
        in.candidates = { pinned_on("m1", "m1/g0", 10 * GB) };
        const admission_result r = decide_admission(in);
        check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_NO_CANDIDATE && r.blocked_on == "m2",
              "override excludes every candidate", r);
    }

    // a slot the ledger does not know never fits
    {
        admission_input in;
        in.alias      = "m";
        in.candidates = { pinned_on("", "ghost", 1 * GB) };
        const admission_result r = decide_admission(in);
        check(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_CAPACITY && r.blocked_on == "ghost",
              "unknown slot", r);
    }

    printf("test-router-admission: OK\n");
    return 0;
}
