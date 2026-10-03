#pragma once

// Hold leases (spec §6): an outside orchestrator says "keep this model resident" for a while.
// A held model is never an eviction victim, never idle-unloaded and never yielded to the
// board queue. A lease renews by holding again with it; an expired lease drops silently.
//
// No locking and no clock of its own: the router guards the table with its mutex and passes
// `now`, so expiry is table-testable with literals.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct router_hold {
    std::string lease;
    std::string model; // canonical model name (a group: its spine)
    std::string owner; // free text, for listings
    int64_t     expires_ms = 0;
};

class router_holds {
  public:
    // A new lease (lease == "") or the renewal of `lease`. Returns the lease, or "" with `err`
    // set: ttl_ms <= 0, an unknown or expired lease, or a lease held for another model.
    std::string hold(const std::string & model, int64_t ttl_ms, const std::string & owner,
                     const std::string & lease, int64_t now_ms, std::string & err);

    // false if the lease is unknown (or already expired)
    bool release(const std::string & lease, int64_t now_ms);

    bool is_held(const std::string & model, int64_t now_ms) const;

    // live leases only
    std::vector<router_hold> list(int64_t now_ms) const;

    // forget expired leases
    void prune(int64_t now_ms);

  private:
    std::map<std::string, router_hold> leases; // lease -> hold
    uint64_t                           counter = 0;
};
