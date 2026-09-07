// MAD-LAB 2026-09-07: unit tests for Leviathan/Chen rejection-sampling
// verification of speculative drafts (common_spec_verify_step).
//
// The whole point of the routine is a distributional identity, so the tests are
// Monte-Carlo:
//
//   (a) the emitted token is distributed exactly as the target p, for any draft
//       distribution q the draft token was actually sampled from -- including a
//       q that is zero outside a small candidate set (the DFlash2 selector
//       lattice case);
//   (b) the acceptance rate is sum_x min(p(x), q(x)), which is what makes this
//       worth doing at all: the legacy match rule only reaches sum_x p(x)q(x);
//   (c) with a greedy (one-hot) target the routine degenerates to the exact
//       match rule, so the temperature-0 path is unchanged.

#include "sampling.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static llama_token sample_from(const std::vector<float> & q, std::mt19937 & rng) {
    const float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    float acc = 0.0f;
    for (size_t i = 0; i < q.size(); ++i) {
        acc += q[i];
        if (acc >= r) {
            return (llama_token) i;
        }
    }
    return (llama_token) (q.size() - 1);
}

// dense p over the whole toy vocab -> the candidate array the sampler chain would
// have produced (entries with p == 0 are dropped, exactly as top-k/top-p would)
static std::vector<llama_token_data> make_p(const std::vector<float> & p) {
    std::vector<llama_token_data> out;
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] > 0.0f) {
            out.push_back({ (llama_token) i, std::log(p[i]), p[i] });
        }
    }
    return out;
}

// sparse q, the shape the drafter hands the verifier
static std::vector<llama_token_data> make_q(const std::vector<float> & q) {
    std::vector<llama_token_data> out;
    for (size_t i = 0; i < q.size(); ++i) {
        if (q[i] > 0.0f) {
            out.push_back({ (llama_token) i, std::log(q[i]), q[i] });
        }
    }
    return out;
}

struct run_result {
    std::vector<double> emp;   // empirical output distribution
    double              acc;   // empirical acceptance rate
};

static run_result run(const std::vector<float> & p, const std::vector<float> & q, int n_iter, uint32_t seed) {
    const auto p_tmpl = make_p(p);
    const auto q_sp   = make_q(q);

    std::mt19937 rng_draft(seed);
    std::mt19937 rng_verif(seed + 1);

    run_result res;
    res.emp.assign(p.size(), 0.0);
    res.acc = 0.0;

    std::vector<llama_token_data> p_buf;

    for (int it = 0; it < n_iter; ++it) {
        const llama_token x = sample_from(q, rng_draft);

        // fresh copy: common_spec_verify_step turns p into the residual in place
        p_buf = p_tmpl;
        llama_token_data_array p_arr = { p_buf.data(), p_buf.size(), -1, false };

        llama_token out = -1;
        const bool accepted = common_spec_verify_step(p_arr, q_sp, x, rng_verif, out);

        if (accepted) {
            res.acc += 1.0;
            if (out != x) {
                fprintf(stderr, "FAIL: accepted but emitted %d != draft %d\n", out, x);
                exit(1);
            }
        }

        if (out < 0 || (size_t) out >= p.size()) {
            fprintf(stderr, "FAIL: out of range token %d\n", out);
            exit(1);
        }
        res.emp[out] += 1.0;
    }

    for (auto & v : res.emp) {
        v /= n_iter;
    }
    res.acc /= n_iter;

    return res;
}

static void check_case(const char * name,
                       const std::vector<float> & p,
                       const std::vector<float> & q,
                       int n_iter,
                       double tol) {
    const auto res = run(p, q, n_iter, 1234);

    double sum_min = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        sum_min += std::min((double) p[i], (double) q[i]);
    }

    // legacy match-based rate, for the record: P(target draw == draft draw)
    double match = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        match += (double) p[i] * (double) q[i];
    }

    printf("%-28s accept=%.4f (expected %.4f, match rule would give %.4f)\n",
           name, res.acc, sum_min, match);

    bool ok = true;

    if (std::fabs(res.acc - sum_min) > tol) {
        fprintf(stderr, "FAIL[%s]: acceptance %.4f != sum min(p,q) %.4f\n", name, res.acc, sum_min);
        ok = false;
    }

    for (size_t i = 0; i < p.size(); ++i) {
        if (std::fabs(res.emp[i] - (double) p[i]) > tol) {
            fprintf(stderr, "FAIL[%s]: output p[%zu] = %.4f, expected %.4f\n", name, i, res.emp[i], (double) p[i]);
            ok = false;
        }
    }

    if (!ok) {
        exit(1);
    }
}

int main() {
    const int    n_iter = 400000;
    const double tol    = 0.005;

    // 1. q close to p, full support: the regime a good draft head is in, where
    //    rejection sampling gains the most over the match rule
    check_case("close q, full support",
               { 0.40f, 0.25f, 0.15f, 0.10f, 0.06f, 0.04f },
               { 0.35f, 0.30f, 0.15f, 0.11f, 0.05f, 0.04f },
               n_iter, tol);

    // 2. q is sparse: zero outside a 3-token candidate set. This is the DFlash2
    //    selector-lattice case -- q restricted to the top-k, everything else 0.
    check_case("sparse q (top-3 lattice)",
               { 0.30f, 0.25f, 0.20f, 0.15f, 0.10f },
               { 0.50f, 0.30f, 0.20f, 0.00f, 0.00f },
               n_iter, tol);

    // 3. q proposes tokens the target's sampler chain truncated away entirely
    //    (p == 0 there): those are always rejected, and the identity must hold.
    check_case("q outside p's support",
               { 0.60f, 0.40f, 0.00f, 0.00f },
               { 0.25f, 0.25f, 0.25f, 0.25f },
               n_iter, tol);

    // 4. q == p: everything is accepted and the output is p
    check_case("q == p",
               { 0.50f, 0.30f, 0.20f },
               { 0.50f, 0.30f, 0.20f },
               n_iter, tol);

    // 5. adversarially bad q
    check_case("bad q",
               { 0.90f, 0.05f, 0.05f },
               { 0.05f, 0.05f, 0.90f },
               n_iter, tol);

    // 6. GREEDY / temperature-0 target: p is one-hot. The routine must degenerate
    //    to the exact-match rule -- accept iff the drafted token IS the argmax,
    //    and otherwise emit the argmax. That is what keeps the temp-0 path
    //    behaviourally identical to the legacy verifier.
    {
        const std::vector<float> p = { 0.0f, 1.0f, 0.0f, 0.0f };
        const std::vector<float> q = { 0.25f, 0.25f, 0.25f, 0.25f };

        const auto p_tmpl = make_p(p);
        const auto q_sp   = make_q(q);

        std::mt19937 rng(7);
        std::vector<llama_token_data> p_buf;

        int n_acc = 0;
        for (llama_token x = 0; x < 4; ++x) {
            for (int it = 0; it < 1000; ++it) {
                p_buf = p_tmpl;
                llama_token_data_array p_arr = { p_buf.data(), p_buf.size(), -1, false };

                llama_token out = -1;
                const bool accepted = common_spec_verify_step(p_arr, q_sp, x, rng, out);

                if (accepted != (x == 1)) {
                    fprintf(stderr, "FAIL[greedy]: draft %d accepted=%d, expected %d\n", x, (int) accepted, (int) (x == 1));
                    return 1;
                }
                if (out != 1) {
                    fprintf(stderr, "FAIL[greedy]: emitted %d, expected the argmax 1\n", out);
                    return 1;
                }
                n_acc += accepted ? 1 : 0;
            }
        }
        printf("%-28s accept=%.4f (expected %.4f)\n", "greedy target (one-hot p)", n_acc / 4000.0, 0.25);
    }

    printf("OK\n");

    return 0;
}
