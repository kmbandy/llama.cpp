#include "server-router-holds.h"

#include <cstdio>
#include <iterator>
#include <random>

static std::string new_lease_id(uint64_t counter) {
    static std::random_device rd;
    char buf[48];
    snprintf(buf, sizeof(buf), "hold-%08x%08x-%llu", (unsigned) rd(), (unsigned) rd(), (unsigned long long) counter);
    return buf;
}

std::string router_holds::hold(const std::string & model, int64_t ttl_ms, const std::string & owner,
                               const std::string & lease, int64_t now_ms, std::string & err) {
    prune(now_ms);
    if (model.empty()) {
        err = "model is required";
        return "";
    }
    if (ttl_ms <= 0) {
        err = "ttl_s must be positive";
        return "";
    }
    if (!lease.empty()) {
        auto it = leases.find(lease);
        if (it == leases.end()) {
            err = "unknown or expired lease '" + lease + "'";
            return "";
        }
        if (it->second.model != model) {
            err = "lease '" + lease + "' holds model '" + it->second.model + "', not '" + model + "'";
            return "";
        }
        it->second.expires_ms = now_ms + ttl_ms;
        if (!owner.empty()) {
            it->second.owner = owner;
        }
        return lease;
    }
    router_hold h;
    h.lease      = new_lease_id(++counter);
    h.model      = model;
    h.owner      = owner;
    h.expires_ms = now_ms + ttl_ms;
    leases[h.lease] = h;
    return h.lease;
}

bool router_holds::release(const std::string & lease, int64_t now_ms) {
    prune(now_ms);
    return leases.erase(lease) > 0;
}

bool router_holds::is_held(const std::string & model, int64_t now_ms) const {
    for (const auto & [_, h] : leases) {
        if (h.model == model && h.expires_ms > now_ms) {
            return true;
        }
    }
    return false;
}

std::vector<router_hold> router_holds::list(int64_t now_ms) const {
    std::vector<router_hold> out;
    for (const auto & [_, h] : leases) {
        if (h.expires_ms > now_ms) {
            out.push_back(h);
        }
    }
    return out;
}

void router_holds::prune(int64_t now_ms) {
    for (auto it = leases.begin(); it != leases.end();) {
        it = it->second.expires_ms <= now_ms ? leases.erase(it) : std::next(it);
    }
}
