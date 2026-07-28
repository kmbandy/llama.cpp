#pragma once
#include <vector>
#include <cstdint>
#include <shared_mutex>
namespace wp {
struct ExpertRef { int layer; int expert; };
class RouterPredictor {
public:
    void set_router(int layer, const float* W, int n_expert, int n_embd);
    bool has_router(int layer) const;
    // Append top-M experts for each target layer from_layer+1..from_layer+K
    // (that has a router and is < n_layer) to out. Plain top-M of W[T].h.
    void predict(const float* h, int from_layer, int K, int M,
                 int n_layer, std::vector<ExpertRef>& out, float min_conf = 0.0f) const;
private:
    bool has_router_locked_(int layer) const;   // caller holds mu_ (shared or unique)
public:
    int n_expert() const { std::shared_lock<std::shared_mutex> lk(mu_); return n_expert_; }
    int n_embd()   const { std::shared_lock<std::shared_mutex> lk(mu_); return n_embd_; }
private:
    struct Router { std::vector<float> W; }; // [n_expert*n_embd] row-major, empty=unset
    std::vector<Router> routers_;            // indexed by layer
    int n_expert_ = 0, n_embd_ = 0;
    // set_router() is called lazily from the EVAL thread as each layer's router
    // weight is first seen, and it RESIZES routers_. predict() may run on the
    // async host-prefetch worker. A vector reallocation under a concurrent
    // reader is undefined behaviour, so the two are serialised: writers take
    // the unique lock, readers the shared one. Contention is negligible --
    // set_router fires at most once per layer for the whole run.
    mutable std::shared_mutex mu_;
};
} // namespace wp
