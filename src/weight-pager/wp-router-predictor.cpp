#include "wp-router-predictor.h"
#include <algorithm>
#include <cmath>
namespace wp {
void RouterPredictor::set_router(int layer, const float* W, int n_expert, int n_embd) {
    if (layer < 0 || W == nullptr || n_expert <= 0 || n_embd <= 0) return;
    if ((int) routers_.size() <= layer) routers_.resize(layer + 1);
    n_expert_ = n_expert; n_embd_ = n_embd;
    routers_[layer].W.assign(W, W + (size_t) n_expert * n_embd);
}
bool RouterPredictor::has_router(int layer) const {
    return layer >= 0 && layer < (int) routers_.size() && !routers_[layer].W.empty();
}
void RouterPredictor::predict(const float* h, int from_layer, int K, int M,
                              int n_layer, std::vector<ExpertRef>& out, float min_conf) const {
    if (h == nullptr || K <= 0 || M <= 0) return;
    std::vector<std::pair<float,int>> logits((size_t) n_expert_);
    for (int d = 1; d <= K; ++d) {
        const int T = from_layer + d;
        if (T >= n_layer || !has_router(T)) continue;
        const float* W = routers_[T].W.data();
        for (int e = 0; e < n_expert_; ++e) {
            const float* w = W + (size_t) e * n_embd_;
            float s = 0.0f;
            for (int j = 0; j < n_embd_; ++j) s += w[j] * h[j];
            logits[(size_t) e] = { s, e };
        }
        const int m = std::min(M, n_expert_);
        std::partial_sort(logits.begin(), logits.begin() + m, logits.end(),
                          [](const std::pair<float,int>& a, const std::pair<float,int>& b){ return a.first > b.first; });
        if (min_conf > 0.0f) {
            // Softmax confidence gate: emit an expert only if its routing
            // probability (softmax over ALL experts' logits) clears min_conf.
            // Peaked layers pass their top experts; flat/uncertain layers pass
            // few or none, so the prefetcher fetches only high-confidence experts.
            // logits[0] is the max after partial_sort; top-m are sorted desc, so
            // once p drops below the threshold every later one does too.
            const float maxl = logits[0].first;
            double denom = 0.0;
            for (int e = 0; e < n_expert_; ++e) denom += std::exp((double)(logits[(size_t) e].first - maxl));
            if (denom <= 0.0) denom = 1.0;
            for (int i = 0; i < m; ++i) {
                const double p = std::exp((double)(logits[(size_t) i].first - maxl)) / denom;
                if (p < (double) min_conf) break;
                out.push_back(ExpertRef{ T, logits[(size_t) i].second });
            }
        } else {
            for (int i = 0; i < m; ++i) out.push_back(ExpertRef{ T, logits[(size_t) i].second });
        }
    }
}
} // namespace wp
