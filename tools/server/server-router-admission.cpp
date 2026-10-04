#include "server-router-admission.h"

#include "server-router-policy.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <set>
#include <utility>

const char * admission_priority_str(admission_priority p) {
    switch (p) {
        case ADMISSION_PRIORITY_HIGHEST: return "highest";
        case ADMISSION_PRIORITY_LOWEST:  return "lowest";
        default:                         return "middle";
    }
}

std::optional<admission_priority> admission_priority_parse(const std::string & s) {
    if (s == "highest") {
        return ADMISSION_PRIORITY_HIGHEST;
    }
    if (s == "middle") {
        return ADMISSION_PRIORITY_MIDDLE;
    }
    if (s == "lowest") {
        return ADMISSION_PRIORITY_LOWEST;
    }
    return std::nullopt;
}

const char * admission_verdict_str(admission_verdict v) {
    switch (v) {
        case ADMISSION_ADMIT:            return "admit";
        case ADMISSION_EVICT_THEN_ADMIT: return "evict_then_admit";
        default:                         return "queue";
    }
}

const char * admission_block_str(admission_block b) {
    switch (b) {
        case ADMISSION_BLOCK_CLAIM:        return "claim";
        case ADMISSION_BLOCK_PINNED:       return "pinned";
        case ADMISSION_BLOCK_HELD:         return "held";
        case ADMISSION_BLOCK_BUSY:         return "busy";
        case ADMISSION_BLOCK_CAPACITY:     return "capacity";
        case ADMISSION_BLOCK_NO_CANDIDATE: return "no_candidate";
        default:                           return "none";
    }
}

const char * admission_notify_kind_str(admission_notify_kind k) {
    return k == ADMISSION_NOTIFY_YIELD ? "yield" : "wait";
}

namespace {

// One eviction unit: a model on its own, or a whole group (spine + workers).
struct admission_unit {
    std::string                    name;
    std::map<std::string, int64_t> vram; // slot -> bytes (present even at 0)
    std::map<std::string, int64_t> ram;  // machine -> bytes (present even at 0)
    std::set<std::string>          exclusive_slots;
    int64_t                        last_used = 0;
    bool                           busy      = false;
    bool                           pinned    = false;
    bool                           held      = false;
};

evict_resident as_evict(const admission_unit & u) {
    return { u.name, u.last_used, u.pinned, u.busy ? 1 : 0, u.held };
}

struct admission_eval {
    bool                                 fits = false;
    std::vector<std::string>             slots;
    std::vector<std::string>             resources;
    std::vector<const admission_claim *> foreign;
    std::vector<std::string>             victims;
    int                                  busy_victims  = 0;
    int64_t                              newest_victim = 0;
    int64_t                              free_vram     = 0; // the tightest chosen slot, before evictions
    admission_block                      blocked = ADMISSION_BLOCK_NONE;
    std::string                          blocked_by;
    std::string                          blocked_on;
    bool                                 blocked_on_ram = false;
};

void add_need(std::vector<std::pair<std::string, int64_t>> & out, const std::string & key, int64_t bytes) {
    for (auto & e : out) {
        if (e.first == key) {
            e.second += std::max<int64_t>(0, bytes);
            return;
        }
    }
    out.push_back({ key, std::max<int64_t>(0, bytes) });
}

void add_unique(std::vector<std::string> & out, const std::string & s) {
    if (!s.empty() && std::find(out.begin(), out.end(), s) == out.end()) {
        out.push_back(s);
    }
}

std::map<std::string, admission_unit> build_units(const admission_input & in) {
    std::map<std::string, admission_unit> units;
    for (const auto & r : in.residents) {
        const std::string key = r.group.empty() ? r.name : r.group;
        // the load itself and its own group are never victims
        if (r.name == in.alias || key == in.alias || (!in.group.empty() && key == in.group)) {
            continue;
        }
        admission_unit & u = units[key];
        u.name = key;
        for (const auto & v : r.vram) {
            u.vram[v.slot] += std::max<int64_t>(0, v.bytes);
            if (r.exclusive) {
                u.exclusive_slots.insert(v.slot);
            }
        }
        for (const auto & m : r.ram) {
            u.ram[m.machine] += std::max<int64_t>(0, m.bytes);
        }
        u.last_used = std::max(u.last_used, r.last_used);
        u.busy      = u.busy || r.busy;
        u.pinned    = u.pinned || r.pinned;
        u.held      = u.held || r.held;
    }
    return units;
}

void set_block(admission_eval & ev, const evict_blocker & b, const std::string & on, bool on_ram) {
    ev.blocked        = b.pinned ? ADMISSION_BLOCK_PINNED : b.held ? ADMISSION_BLOCK_HELD : ADMISSION_BLOCK_BUSY;
    ev.blocked_by     = b.name;
    ev.blocked_on     = on;
    ev.blocked_on_ram = on_ram;
}

admission_eval evaluate(const admission_input & in, const admission_candidate & cand,
                        const std::map<std::string, admission_unit> & units, const std::vector<evict_resident> & order) {
    admission_eval ev;
    const bool highest = in.priority == ADMISSION_PRIORITY_HIGHEST;

    // the footprint, summed per slot / machine, in first-seen order
    std::vector<std::pair<std::string, int64_t>> vram_need;
    std::vector<std::pair<std::string, int64_t>> ram_need;
    for (const auto & v : cand.vram) {
        add_need(vram_need, v.slot, v.bytes);
    }
    for (const auto & m : cand.ram) {
        add_need(ram_need, m.machine, m.bytes);
    }

    std::map<std::string, int64_t> free_vram;
    std::map<std::string, int64_t> free_ram;
    bool first_slot = true;
    for (const auto & e : vram_need) {
        const std::string & slot = e.first; // not a structured binding: C++17 lambdas cannot capture those
        ev.slots.push_back(slot);
        auto s = std::find_if(in.slots.begin(), in.slots.end(), [&](const admission_slot & x) { return x.id == slot; });
        if (s == in.slots.end()) {
            // a slot the ledger does not know never fits
            ev.blocked    = ADMISSION_BLOCK_CAPACITY;
            ev.blocked_on = slot;
            return ev;
        }
        free_vram[slot] = s->free;
        ev.free_vram    = first_slot ? s->free : std::min(ev.free_vram, s->free);
        first_slot      = false;
        add_unique(ev.resources, s->resource);
    }
    for (const auto & e : ram_need) {
        const std::string & machine = e.first;
        auto m = std::find_if(in.machines.begin(), in.machines.end(), [&](const admission_machine & x) { return x.name == machine; });
        free_ram[machine] = m == in.machines.end() ? -1 : m->free_ram;
        if (m != in.machines.end() && e.second > 0) {
            add_unique(ev.resources, m->resource);
        }
    }
    for (const auto & c : in.claims) {
        if (!c.is_router && std::find(ev.resources.begin(), ev.resources.end(), c.resource) != ev.resources.end()) {
            ev.foreign.push_back(&c);
        }
    }

    std::set<std::string> taken;
    // a victim frees its whole footprint: every slot and every machine's RAM it spans
    auto take = [&](const admission_unit & u) {
        taken.insert(u.name);
        ev.victims.push_back(u.name);
        ev.busy_victims += u.busy ? 1 : 0;
        ev.newest_victim = std::max(ev.newest_victim, u.last_used);
        for (const auto & [slot, bytes] : u.vram) {
            auto it = free_vram.find(slot);
            if (it != free_vram.end()) {
                it->second += bytes;
            }
        }
        for (const auto & [machine, bytes] : u.ram) {
            auto it = free_ram.find(machine);
            if (it != free_ram.end() && it->second >= 0) {
                it->second += bytes;
            }
        }
    };
    // residents still in the way on a slot / machine, for naming the blocker
    auto remaining = [&](bool ram, const std::string & key) {
        std::vector<evict_resident> out;
        for (const auto & [name, u] : units) {
            if (!taken.count(name) && (ram ? u.ram.count(key) : u.vram.count(key))) {
                out.push_back(as_evict(u));
            }
        }
        return out;
    };
    // evict in policy order (idle LRU, then busy LRU at highest) until `free` reaches `need`
    auto make_room = [&](bool ram, const std::string & key, const int64_t & free, int64_t need) {
        for (const auto & r : order) {
            if (free >= need) {
                break;
            }
            const admission_unit & u = units.at(r.name);
            if (taken.count(r.name) || !(ram ? u.ram.count(key) : u.vram.count(key))) {
                continue;
            }
            take(u);
        }
        if (free >= need) {
            return true;
        }
        if (const auto b = evict_find_blocker(remaining(ram, key), highest)) {
            set_block(ev, *b, key, ram);
        } else {
            ev.blocked        = ADMISSION_BLOCK_CAPACITY;
            ev.blocked_on     = key;
            ev.blocked_on_ram = ram;
        }
        return false;
    };

    // residents that must go whatever the free space: everyone on the slots of an exclusive
    // load, and an exclusive resident on any slot this load uses
    for (const auto & [slot, need] : vram_need) {
        std::vector<evict_resident> must;
        for (const auto & [name, u] : units) {
            if (!taken.count(name) && u.vram.count(slot) && (in.exclusive || u.exclusive_slots.count(slot))) {
                must.push_back(as_evict(u));
            }
        }
        if (const auto b = evict_find_blocker(must, highest)) {
            set_block(ev, *b, slot, false);
            return ev;
        }
        // idle before busy, LRU first, so the eviction order is the policy order
        for (const auto & r : evict_pick_lru(must, highest)) {
            take(units.at(r.name));
        }
    }

    // an exclusive load owns its cards whole; its VRAM is not gated (today's behavior)
    if (!in.exclusive) {
        for (const auto & [slot, need] : vram_need) {
            if (!make_room(false, slot, free_vram[slot], need + in.margin_bytes)) {
                return ev;
            }
        }
    }
    for (const auto & [machine, need] : ram_need) {
        if (need <= 0 || free_ram[machine] < 0) {
            continue; // nothing needed, or unknown free RAM: never gates
        }
        if (!make_room(true, machine, free_ram[machine], need)) {
            return ev;
        }
    }
    ev.fits = true;
    return ev;
}

// fewest busy victims, then fewest victims, then least recently used victims, then most free VRAM
bool better(const admission_eval & a, const admission_eval & b) {
    if (a.busy_victims != b.busy_victims) {
        return a.busy_victims < b.busy_victims;
    }
    if (a.victims.size() != b.victims.size()) {
        return a.victims.size() < b.victims.size();
    }
    if (a.newest_victim != b.newest_victim) {
        return a.newest_victim < b.newest_victim;
    }
    return a.free_vram > b.free_vram;
}

} // namespace

admission_result decide_admission(const admission_input & in) {
    admission_result res;
    const auto units = build_units(in);
    std::vector<evict_resident> all;
    for (const auto & [name, u] : units) {
        all.push_back(as_evict(u));
    }
    const std::vector<evict_resident> order = evict_pick_lru(all, in.priority == ADMISSION_PRIORITY_HIGHEST);

    std::vector<admission_eval> evs(in.candidates.size());
    int best         = -1; // fits, nothing claimed
    int best_claimed = -1; // a foreign claim is in the way
    int blocked      = -1; // in the way of residents / capacity
    for (size_t i = 0; i < in.candidates.size(); ++i) {
        if (!in.machine.empty() && in.candidates[i].machine != in.machine) {
            continue;
        }
        evs[i] = evaluate(in, in.candidates[i], units, order);
        const admission_eval & ev = evs[i];
        if (!ev.foreign.empty()) {
            // report the claimed candidate that would otherwise fit best
            if (best_claimed < 0 || (ev.fits && !evs[best_claimed].fits) ||
                    (ev.fits == evs[best_claimed].fits && better(ev, evs[best_claimed]))) {
                best_claimed = (int) i;
            }
        } else if (ev.fits) {
            if (best < 0 || better(ev, evs[best])) {
                best = (int) i;
            }
        } else if (blocked < 0 || (evs[blocked].blocked == ADMISSION_BLOCK_CAPACITY && ev.blocked != ADMISSION_BLOCK_CAPACITY)) {
            // prefer naming a resident in the way over a plain capacity shortfall
            blocked = (int) i;
        }
    }

    if (best >= 0) {
        const admission_eval & ev = evs[best];
        res.verdict              = ev.victims.empty() ? ADMISSION_ADMIT : ADMISSION_EVICT_THEN_ADMIT;
        res.candidate            = (size_t) best;
        res.slots                = ev.slots;
        res.victims              = ev.victims;
        res.board_claims_to_take = ev.resources;
        return res;
    }

    res.verdict    = ADMISSION_QUEUE;
    res.queue_head = in.priority == ADMISSION_PRIORITY_HIGHEST;
    if (best_claimed >= 0) {
        const admission_eval & ev = evs[best_claimed];
        res.candidate  = (size_t) best_claimed;
        res.slots      = ev.slots;
        res.blocked    = ADMISSION_BLOCK_CLAIM;
        res.blocked_by = ev.foreign.front()->holder;
        res.blocked_on = ev.foreign.front()->resource;
        if (in.priority != ADMISSION_PRIORITY_LOWEST) {
            const admission_notify_kind kind = in.priority == ADMISSION_PRIORITY_HIGHEST ? ADMISSION_NOTIFY_YIELD : ADMISSION_NOTIFY_WAIT;
            for (const admission_claim * c : ev.foreign) {
                const bool seen = std::any_of(res.notify.begin(), res.notify.end(),
                                              [&](const admission_notify & n) { return n.claim_id == c->claim_id; });
                if (!seen) {
                    res.notify.push_back({ c->claim_id, c->holder, kind });
                }
            }
        }
        return res;
    }
    if (blocked >= 0) {
        const admission_eval & ev = evs[blocked];
        res.candidate      = (size_t) blocked;
        res.slots          = ev.slots;
        res.blocked        = ev.blocked;
        res.blocked_by     = ev.blocked_by;
        res.blocked_on     = ev.blocked_on;
        res.blocked_on_ram = ev.blocked_on_ram;
        return res;
    }
    res.blocked    = ADMISSION_BLOCK_NO_CANDIDATE;
    res.blocked_on = in.machine;
    return res;
}

std::vector<admission_candidate> admission_pool_candidates(const std::vector<admission_slot> & slots, int64_t vram, int64_t ram) {
    std::vector<admission_candidate> out;
    for (const auto & s : slots) {
        admission_candidate c;
        c.machine = s.machine;
        c.vram    = { { s.id, vram } };
        if (ram > 0) {
            c.ram = { { s.machine, ram } };
        }
        out.push_back(c);
    }
    return out;
}

// ---- Pools and replicas -------------------------------------------------------------------

router_pool_spec router_pool_parse(const std::string & placement, const std::string & replicas, const std::string & pool_gpus) {
    router_pool_spec out;
    const auto strip = [](const std::string & s) {
        size_t b = 0;
        size_t e = s.size();
        while (b < e && isspace((unsigned char) s[b])) { b++; }
        while (e > b && isspace((unsigned char) s[e - 1])) { e--; }
        return s.substr(b, e - b);
    };
    const std::string pl = strip(placement);
    if (pl.empty() || pl == "pinned") {
        out.pool = false;
    } else if (pl == "any") {
        out.pool = true;
    } else {
        out.err = "placement must be 'pinned' or 'any', got '" + pl + "'";
        return out;
    }
    const std::string rep = strip(replicas);
    if (!rep.empty()) {
        char * end = nullptr;
        const long n = strtol(rep.c_str(), &end, 10);
        if (end == rep.c_str() || *end != '\0' || n < 1 || n > 64) {
            out.err = "replicas must be an integer from 1 to 64, got '" + rep + "'";
            return out;
        }
        out.replicas = (int) n;
    }
    size_t pos = 0;
    while (pos <= pool_gpus.size()) {
        size_t comma = pool_gpus.find(',', pos);
        if (comma == std::string::npos) {
            comma = pool_gpus.size();
        }
        const std::string g = strip(pool_gpus.substr(pos, comma - pos));
        if (!g.empty()) {
            out.gpus.push_back(g);
        }
        pos = comma + 1;
    }
    if (!out.pool && out.replicas > 1) {
        out.err = "replicas > 1 needs placement = any";
    } else if (!out.pool && !out.gpus.empty()) {
        out.err = "pool-gpus needs placement = any";
    }
    return out;
}

std::string router_replica_name(const std::string & alias, int k) {
    return k <= 1 ? alias : alias + "~r" + std::to_string(k);
}

bool router_replica_split(const std::string & name, std::string & alias, int & k) {
    const size_t pos = name.rfind("~r");
    if (pos == std::string::npos || pos == 0 || pos + 2 >= name.size()) {
        alias = name;
        k     = 1;
        return false;
    }
    int n = 0;
    for (size_t i = pos + 2; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') {
            alias = name;
            k     = 1;
            return false;
        }
        n = n * 10 + (name[i] - '0');
        if (n > 1000000) {
            break;
        }
    }
    if (n < 2) {
        alias = name;
        k     = 1;
        return false;
    }
    alias = name.substr(0, pos);
    k     = n;
    return true;
}

bool router_replica_name_reserved(const std::string & name) {
    const size_t pos = name.rfind("~r");
    if (pos == std::string::npos || pos == 0 || pos + 2 >= name.size()) {
        return false;
    }
    for (size_t i = pos + 2; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return false;
        }
    }
    return true;
}

bool router_replica_slot_taken(const std::string & slot, const std::vector<std::string> & held) {
    return std::find(held.begin(), held.end(), slot) != held.end();
}

std::vector<std::string> router_pool_slots(const std::vector<std::string> & all, const std::function<bool(const std::string &)> & usable,
                                           const std::set<std::string> & taken) {
    std::vector<std::string> out;
    for (const auto & s : all) {
        if (taken.count(s) == 0 && (!usable || usable(s))) {
            out.push_back(s);
        }
    }
    return out;
}

router_replica_choice router_replica_choose(const std::vector<router_replica_state> & states, bool can_add) {
    router_replica_choice c;
    int  least   = -1;
    int  loading = -1;
    int  down    = -1;
    for (size_t i = 0; i < states.size(); ++i) {
        const auto & s = states[i];
        if (s.status == ROUTER_REPLICA_READY) {
            if (least < 0 || s.inflight < states[least].inflight) {
                least = (int) i;
            }
        } else if (s.status == ROUTER_REPLICA_LOADING) {
            loading = loading < 0 ? (int) i : loading;
        } else {
            down = down < 0 ? (int) i : down;
        }
    }
    if (least >= 0) {
        c.use = least;
        if (states[least].inflight > 0 && loading < 0) {
            if (down >= 0) {
                c.grow = down;
            } else if (can_add) {
                c.grow_new = true;
            }
        }
        return c;
    }
    if (loading >= 0) {
        c.use = loading;
    } else if (down >= 0) {
        c.use = down;
    } else if (can_add) {
        c.use_new = true;
    }
    return c;
}
