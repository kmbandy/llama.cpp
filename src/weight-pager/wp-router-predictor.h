#pragma once
#include <vector>
#include <cstdint>
namespace wp {
struct ExpertRef { int layer; int expert; };
class RouterPredictor {
public:
    void set_router(int layer, const float* W, int n_expert, int n_embd);
    bool has_router(int layer) const;
    // Append top-M experts for each target layer from_layer+1..from_layer+K
    // (that has a router and is < n_layer) to out. Plain top-M of W[T].h.
    void predict(const float* h, int from_layer, int K, int M,
                 int n_layer, std::vector<ExpertRef>& out) const;
    int n_expert() const { return n_expert_; }
    int n_embd()   const { return n_embd_; }
private:
    struct Router { std::vector<float> W; }; // [n_expert*n_embd] row-major, empty=unset
    std::vector<Router> routers_;            // indexed by layer
    int n_expert_ = 0, n_embd_ = 0;
};
} // namespace wp
