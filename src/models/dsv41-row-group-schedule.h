#pragma once

// WP_SPINE_ROW_GROUPS=2 (S5): emission order of the two-lane trunk ping-pong.
//
// Lane 0 (A) owns rows [0,k), lane 1 (B) owns rows [k,R). Each layer is split into
// PRE(g,l) (attention, router, issue, shared expert) and POST(g,l) (wait, add,
// hc_post, cvec). The schedule runs lane A one layer ahead of lane B:
//
//   PRE(A,0) PRE(B,0)
//   for l: POST(A,l) PRE(A,l+1)  POST(B,l) PRE(B,l+1)
//
// so issue(A,l+1) is emitted before wait(B,l), and the open dispatches never
// exceed two. The graph builder expands the ggml graph after each step, so the
// node order equals this order. T3 runs the same schedule with fake dispatch ops.

#include <vector>

enum class dsv41_rg_kind { PRE, POST };

struct dsv41_rg_step {
    dsv41_rg_kind kind;
    int           group; // 0 = A, 1 = B
    int           layer;
};

inline std::vector<dsv41_rg_step> dsv41_row_group_schedule(int n_layer) {
    std::vector<dsv41_rg_step> s;
    if (n_layer <= 0) {
        return s;
    }
    s.push_back({ dsv41_rg_kind::PRE, 0, 0 });
    s.push_back({ dsv41_rg_kind::PRE, 1, 0 });
    for (int l = 0; l < n_layer; ++l) {
        s.push_back({ dsv41_rg_kind::POST, 0, l });
        if (l + 1 < n_layer) {
            s.push_back({ dsv41_rg_kind::PRE, 0, l + 1 });
        }
        s.push_back({ dsv41_rg_kind::POST, 1, l });
        if (l + 1 < n_layer) {
            s.push_back({ dsv41_rg_kind::PRE, 1, l + 1 });
        }
    }
    return s;
}
