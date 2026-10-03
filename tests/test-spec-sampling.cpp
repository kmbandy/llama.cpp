// CPU-only check of the speculative-sampling accept/residual/bonus math (no model, no GPU).
//
// A Markov "target" and a different Markov "draft" over an 8-token vocab are used. The draft chain is
// sampled from q, verified with common_spec_verify_stochastic(), and the distribution of the emitted
// tokens must equal the target's: the marginal of every output position and the joint of the first two.

#include "sampling.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static constexpr int V = 8;

using dist_t = std::vector<double>;

static dist_t make_dist(std::mt19937 & rng, double sharp) {
    std::uniform_real_distribution<double> u(0.02, 1.0);
    dist_t d(V);
    double s = 0.0;
    for (auto & x : d) { x = std::pow(u(rng), sharp); s += x; }
    for (auto & x : d) { x /= s; }
    return d;
}

static int sample(const dist_t & d, std::mt19937 & rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double t = u(rng), run = 0.0;
    for (int i = 0; i < V; ++i) {
        run += d[i];
        if (t < run) { return i; }
    }
    return V - 1;
}

int main() {
    std::mt19937 rng(1234);

    // P[prev] / Q[prev]: next-token distributions given the previous token (row V = start state)
    std::vector<dist_t> P, Q;
    for (int i = 0; i <= V; ++i) { P.push_back(make_dist(rng, 3.0)); Q.push_back(make_dist(rng, 1.0)); }

    const int    n_draft = 3;
    const int    n_trials = 200000;
    const int    n_out_max = n_draft + 1;

    std::vector<std::vector<double>> marg(n_out_max, std::vector<double>(V, 0.0));
    std::vector<double>              joint2(V*V, 0.0);
    std::vector<double>              n_at(n_out_max, 0.0);
    double n_all_accepted = 0.0;
    double mean_accept = 0.0, n_accept_tests = 0.0;

    for (int t = 0; t < n_trials; ++t) {
        // draft chain sampled from q
        std::vector<llama_token> draft;
        int prev = V;
        for (int i = 0; i < n_draft; ++i) {
            const int x = sample(Q[prev], rng);
            draft.push_back(x);
            prev = x;
        }

        std::vector<llama_token> out;
        const auto target_p = [&](size_t i, std::vector<llama_token_data> & cand) {
            const int pv = i == 0 ? V : out[i - 1]; // history already reported through on_token
            cand.clear();
            for (int k = 0; k < V; ++k) { cand.push_back({ k, 0.0f, (float) P[pv][k] }); }
        };
        const auto q_prob = [&](size_t i, llama_token tok) {
            const int pv = i == 0 ? V : draft[i - 1];
            return Q[pv][tok];
        };
        const auto on_token = [&](size_t, llama_token tok) { out.push_back(tok); };

        std::vector<float> probs;
        // out is filled by on_token; the returned vector must match it
        const auto res = common_spec_verify_stochastic(n_draft, draft.data(), target_p, q_prob, on_token, rng, &probs);
        if (res != out) { std::fprintf(stderr, "result/on_token mismatch\n"); return 1; }
        for (float a : probs) { mean_accept += a; n_accept_tests += 1.0; }

        if ((int) res.size() == n_out_max) { n_all_accepted += 1.0; }
        for (size_t i = 0; i < res.size(); ++i) { marg[i][res[i]] += 1.0; n_at[i] += 1.0; }
        if (res.size() >= 2) { joint2[res[0]*V + res[1]] += 1.0; }
    }

    // Position 0 is always emitted and must follow the target marginal P[V] exactly.
    double max_err = 0.0;

    // position 0: exact target marginal P[V]
    for (int k = 0; k < V; ++k) {
        max_err = std::max(max_err, std::fabs(marg[0][k]/n_at[0] - P[V][k]));
    }

    // Joint law of the first two tokens of the stream. A round that stops after one token is continued with
    // plain target sampling (what the next verify round does), so the stream must follow P[V][a]*P[a][b].
    std::vector<double> joint2_full(V*V, 0.0);
    {
        std::mt19937 rng2(777);
        for (int t = 0; t < n_trials; ++t) {
            std::vector<llama_token> draft;
            int prev = V;
            for (int i = 0; i < n_draft; ++i) { const int x = sample(Q[prev], rng2); draft.push_back(x); prev = x; }
            std::vector<llama_token> out;
            const auto target_p = [&](size_t i, std::vector<llama_token_data> & cand) {
                const int pv = i == 0 ? V : out[i - 1];
                cand.clear();
                for (int k = 0; k < V; ++k) { cand.push_back({ k, 0.0f, (float) P[pv][k] }); }
            };
            const auto q_prob = [&](size_t i, llama_token tok) { return Q[i == 0 ? V : draft[i - 1]][tok]; };
            const auto on_token = [&](size_t, llama_token tok) { out.push_back(tok); };
            auto res = common_spec_verify_stochastic(n_draft, draft.data(), target_p, q_prob, on_token, rng2);
            while (res.size() < 2) {
                res.push_back(sample(P[res.back()], rng2));
            }
            joint2_full[res[0]*V + res[1]] += 1.0;
        }
    }
    double max_err_joint = 0.0;
    for (int a = 0; a < V; ++a) {
        for (int b = 0; b < V; ++b) {
            const double ref = P[V][a] * P[a][b];
            max_err_joint = std::max(max_err_joint, std::fabs(joint2_full[a*V + b]/n_trials - ref));
        }
    }

    // bonus path: with n_draft tokens equal to the target's argmax chain and q == p, everything is accepted
    // and the bonus token must follow p_n
    double max_err_bonus = 0.0;
    {
        std::mt19937 rng3(99);
        std::vector<double> bonus(V, 0.0);
        const int last = 3;
        // only trials whose last draft token equals `last` condition the bonus draw: use more of them
        for (int t = 0; t < 4*n_trials; ++t) {
            std::vector<llama_token> draft;
            int prev = V;
            for (int i = 0; i < n_draft; ++i) { const int x = sample(P[prev], rng3); draft.push_back(x); prev = x; }
            std::vector<llama_token> out;
            const auto target_p = [&](size_t i, std::vector<llama_token_data> & cand) {
                const int pv = i == 0 ? V : draft[i - 1];
                cand.clear();
                for (int k = 0; k < V; ++k) { cand.push_back({ k, 0.0f, (float) P[pv][k] }); }
            };
            const auto q_prob = [&](size_t i, llama_token tok) { return P[i == 0 ? V : draft[i - 1]][tok]; };
            const auto on_token = [&](size_t, llama_token tok) { out.push_back(tok); };
            const auto res = common_spec_verify_stochastic(n_draft, draft.data(), target_p, q_prob, on_token, rng3);
            if ((int) res.size() != n_draft + 1) { std::fprintf(stderr, "q == p must accept every draft token\n"); return 1; }
            if (draft.back() == last) { bonus[res.back()] += 1.0; }
        }
        double n_cond = 0.0;
        for (double b : bonus) { n_cond += b; }
        for (int k = 0; k < V; ++k) {
            max_err_bonus = std::max(max_err_bonus, std::fabs(bonus[k]/n_cond - P[last][k]));
        }
        std::printf("bonus path: %.0f conditioned trials\n", n_cond);
    }

    std::printf("trials=%d  all-accepted=%.3f  mean accept prob=%.3f\n",
            n_trials, n_all_accepted/n_trials, mean_accept/n_accept_tests);
    std::printf("max abs err: pos0 marginal=%.5f  joint(t0,t1)=%.5f  bonus=%.5f\n", max_err, max_err_joint, max_err_bonus);

    const double tol = 0.005;
    if (max_err > tol || max_err_joint > tol || max_err_bonus > tol) {
        std::fprintf(stderr, "FAIL: error above %.3f\n", tol);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
