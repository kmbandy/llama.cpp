#pragma once

// Admission as a pure function (spec §6): given what a load needs, the ledger, the router's
// residents and the board's claims, decide admit / evict-then-admit / queue, with the victims,
// the board claims to take and whom to notify. No I/O and no server_models types: the caller
// gathers the inputs and acts on the result (take claims, evict, wait for the VRAM to be
// confirmed freed, load), so every rule here is table-testable with literals.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

enum admission_priority {
    ADMISSION_PRIORITY_LOWEST  = 0,
    ADMISSION_PRIORITY_MIDDLE  = 1, // default
    ADMISSION_PRIORITY_HIGHEST = 2,
};

// "highest" / "middle" / "lowest"
const char * admission_priority_str(admission_priority p);
std::optional<admission_priority> admission_priority_parse(const std::string & s);

enum admission_verdict {
    ADMISSION_ADMIT            = 0, // fits as is
    ADMISSION_EVICT_THEN_ADMIT = 1, // fits once `victims` are gone
    ADMISSION_QUEUE            = 2, // cannot load now; see blocked / blocked_by
};

const char * admission_verdict_str(admission_verdict v);

// Why a `queue` verdict could not admit.
enum admission_block {
    ADMISSION_BLOCK_NONE         = 0,
    ADMISSION_BLOCK_CLAIM        = 1, // a needed resource is claimed on the board by someone else
    ADMISSION_BLOCK_PINNED       = 2, // a resident that has to go is pinned
    ADMISSION_BLOCK_HELD         = 3, // a resident that has to go is under a hold lease
    ADMISSION_BLOCK_BUSY         = 4, // a resident that has to go has requests in flight (below `highest`)
    ADMISSION_BLOCK_CAPACITY     = 5, // not enough VRAM / RAM even with every evictable resident gone
    ADMISSION_BLOCK_NO_CANDIDATE = 6, // no candidate placement (e.g. machine override excludes them all)
};

const char * admission_block_str(admission_block b);

struct admission_slot_bytes {
    std::string slot; // slot id (ledger_slot_id)
    int64_t     bytes = 0;
};

struct admission_machine_bytes {
    std::string machine; // "" = the router's own machine
    int64_t     bytes = 0;
};

// Ledger view of one GPU slot.
struct admission_slot {
    std::string id;
    std::string machine;
    std::string resource; // board resource name ("gpu:<board name>"); "" = not on the board
    int64_t     free = 0; // ledger free bytes (ledger_free_vram)
};

// Ledger view of one machine's host RAM.
struct admission_machine {
    std::string name;
    std::string resource;      // board resource name for its RAM; "" = not on the board
    int64_t     free_ram = -1; // ledger_free_ram; < 0 = unknown, never gates
};

// One way to place the load: its whole footprint (a group: spine + every worker) as VRAM per
// slot and host RAM per machine. A pinned alias has one candidate; a pool one per slot.
struct admission_candidate {
    std::string                          machine; // where the model itself runs (a machine override matches this)
    std::vector<admission_slot_bytes>    vram;
    std::vector<admission_machine_bytes> ram;
};

// One router-owned process (a model, or a group's spine or worker). Group members share
// `group` (the spine's name) and are evicted together, as one victim named after the group.
struct admission_resident {
    std::string                          name;
    std::string                          group; // "" = not in a group
    std::vector<admission_slot_bytes>    vram;  // what it holds on each slot (0 bytes = present, size unknown)
    std::vector<admission_machine_bytes> ram;   // what it holds in each machine's RAM (0 = present, size unknown)
    int64_t                              last_used = 0;
    bool                                 busy      = false; // requests in flight
    bool                                 pinned    = false;
    bool                                 held      = false; // hold lease
    bool                                 exclusive = false; // owns its slots: anything placed there must evict it
};

// An active board claim on a resource.
struct admission_claim {
    std::string        claim_id;
    std::string        resource;
    std::string        holder;
    bool               is_router = false; // held by the router itself (holder `llama-router`): never blocks
    admission_priority priority  = ADMISSION_PRIORITY_MIDDLE; // the holder's; carried for the caller, no rule reads it
};

struct admission_input {
    std::string                      alias;
    std::string                      group;   // the load's own group ("" = none): its members are never victims
    admission_priority               priority = ADMISSION_PRIORITY_MIDDLE;
    std::string                      machine; // machine override; "" = any
    bool                             exclusive = false; // every other resident on a chosen slot must go; VRAM fit is not gated
    int64_t                          margin_bytes = 0;  // headroom required on top of each slot's need
    std::vector<admission_candidate> candidates;
    std::vector<admission_slot>      slots;
    std::vector<admission_machine>   machines;
    std::vector<admission_resident>  residents;
    std::vector<admission_claim>     claims; // active claims only
};

enum admission_notify_kind {
    ADMISSION_NOTIFY_WAIT  = 0, // `middle`: tell the holder someone is waiting
    ADMISSION_NOTIFY_YIELD = 1, // `highest`: ask the holder to yield (never kills its process)
};

const char * admission_notify_kind_str(admission_notify_kind k);

struct admission_notify {
    std::string           claim_id;
    std::string           holder;
    admission_notify_kind kind = ADMISSION_NOTIFY_WAIT;
};

struct admission_result {
    admission_verdict             verdict = ADMISSION_QUEUE;
    size_t                        candidate = 0;   // index into input.candidates (when one was picked or reported)
    std::vector<std::string>      slots;           // the candidate's slots, in footprint order
    std::vector<std::string>      victims;         // resident names (a group: its group name), eviction order
    std::vector<std::string>      board_claims_to_take; // the candidate's resources
    std::vector<admission_notify> notify;          // queue on a foreign claim only
    bool                          queue_head = false; // queue: at the head (`highest`)
    admission_block               blocked = ADMISSION_BLOCK_NONE;
    std::string                   blocked_by;      // claim holder or resident (group) name; "" for capacity
    std::string                   blocked_on;      // the slot id, machine name or board resource it waits for
    bool                          blocked_on_ram = false; // blocked_on is a machine's host RAM
};

// Spec §6:
//  1. Candidates: those whose machine matches the override (when set).
//  2. A needed resource claimed on the board by someone else -> queue; notify the holder:
//     `yield` at highest, `wait` at middle, nothing at lowest. Queue at the head at highest.
//  3. Victims among residents: idle first (LRU), busy only at highest; pinned and held never;
//     a group member stands for its whole group, whose freed footprint is credited on every
//     slot and machine it spans. The load's own group is never a victim.
//  4. Pick, among candidates that fit with no claim in the way: fewest busy victims, then
//     fewest victims, then least recently used victims, then most free VRAM, then input order.
// Queue verdicts name the first thing in the way (blocked / blocked_by / blocked_on).
admission_result decide_admission(const admission_input & in);

// One candidate per slot (each with `vram` on it and `ram` on its machine): a pool.
std::vector<admission_candidate> admission_pool_candidates(const std::vector<admission_slot> & slots, int64_t vram, int64_t ram);
