// T3: WP_SPINE_ROW_GROUPS (S5) graph order. CPU only, no model.
//
// The two-lane trunk schedule (src/models/dsv41-row-group-schedule.h) is built as
// a real ggml graph with fake issue/wait custom ops that append to a global log.
// Per segment, PRE(g,l) emits issue(g,l) (map_custom1, CPU) and a shared-expert
// stand-in that depends on the issue node; POST(g,l) emits wait(g,l) (map_custom1,
// CPU) and an add. The graph is expanded after each segment, as the trunk builder
// does, and then computed on the ggml CPU backend. The executed log must equal:
//
//   I(A,0) I(B,0)
//   for l: W(A,l) [I(A,l+1)] W(B,l) [I(B,l+1)]
//
// Also checked: issue(A,l+1) precedes wait(B,l), waits run in issue order, and at
// most two dispatches are open at once (k_max_open_dispatches).

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "models/dsv41-row-group-schedule.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct log_entry {
    char kind; // 'I' issue, 'W' wait
    int  group;
    int  layer;
};

std::vector<log_entry> g_log;

// userdata encodes (kind, group, layer) so the callbacks need no allocation
void * encode(char kind, int group, int layer) {
    const intptr_t code = ((kind == 'I' ? 1 : 2) << 16) | (group << 8) | layer;
    return (void *) code;
}

void fake_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int, void * userdata) {
    const intptr_t code = (intptr_t) userdata;
    if (ith == 0) {
        g_log.push_back({ (code >> 16) == 1 ? 'I' : 'W', (int) ((code >> 8) & 0xff), (int) (code & 0xff) });
        std::memcpy(dst->data, a->data, ggml_nbytes(dst));
    }
}

int fail(const char * msg) {
    std::printf("FAIL: %s\n", msg);
    return 1;
}

} // namespace

int main() {
    const int n_layer = 40;
    const int n_elem  = 4;

    ggml_init_params ip = {
        /*.mem_size   =*/ 64u * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return fail("ggml_init");
    }

    ggml_tensor * cur[2] = { nullptr, nullptr };
    ggml_tensor * issued[2] = { nullptr, nullptr };
    ggml_tensor * shexp[2] = { nullptr, nullptr };

    for (int g = 0; g < 2; ++g) {
        cur[g] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elem);
        for (int i = 0; i < n_elem; ++i) {
            ((float *) cur[g]->data)[i] = (float) (g * 10 + i);
        }
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    for (const dsv41_rg_step & s : dsv41_row_group_schedule(n_layer)) {
        const int g = s.group;
        if (s.kind == dsv41_rg_kind::PRE) {
            // issue(g,l): CPU dispatch node on the lane's activations
            issued[g] = ggml_map_custom1(ctx, cur[g], fake_op, 1, encode('I', g, s.layer));
            // shared expert stand-in: GPU op that must wait for the issue (acc with 0*issued)
            ggml_tensor * sh = ggml_scale(ctx, cur[g], 1.0f);
            sh = ggml_add(ctx, sh, ggml_scale(ctx, issued[g], 0.0f));
            shexp[g] = sh;
            cur[g]   = sh; // carries the lane forward for the attention stand-in
            ggml_build_forward_expand(gf, shexp[g]);
        } else {
            // wait(g,l): CPU node joining the dispatch result, then the residual add
            ggml_tensor * w = ggml_map_custom1(ctx, issued[g], fake_op, 1, encode('W', g, s.layer));
            ggml_tensor * out = ggml_add(ctx, shexp[g], w);
            cur[g] = out;
            ggml_build_forward_expand(gf, cur[g]);
        }
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 1);
    const ggml_status st = ggml_backend_graph_compute(cpu, gf);
    ggml_backend_free(cpu);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        return fail("graph compute");
    }

    // expected executed order, written out by hand from the spec
    std::vector<log_entry> want;
    want.push_back({ 'I', 0, 0 });
    want.push_back({ 'I', 1, 0 });
    for (int l = 0; l < n_layer; ++l) {
        want.push_back({ 'W', 0, l });
        if (l + 1 < n_layer) {
            want.push_back({ 'I', 0, l + 1 });
        }
        want.push_back({ 'W', 1, l });
        if (l + 1 < n_layer) {
            want.push_back({ 'I', 1, l + 1 });
        }
    }

    int rc = 0;
    if (g_log.size() != want.size()) {
        std::printf("log size %zu want %zu\n", g_log.size(), want.size());
        rc = fail("executed log length");
    } else {
        for (size_t i = 0; i < want.size(); ++i) {
            if (g_log[i].kind != want[i].kind || g_log[i].group != want[i].group || g_log[i].layer != want[i].layer) {
                std::printf("pos %zu got %c%c%d want %c%c%d\n", i,
                        g_log[i].kind, g_log[i].group == 0 ? 'A' : 'B', g_log[i].layer,
                        want[i].kind, want[i].group == 0 ? 'A' : 'B', want[i].layer);
                rc = fail("executed order differs from A0,B0,WA0,IA1,WB0,IB1,...");
                break;
            }
        }
    }

    // invariants, independent of the hand-written sequence
    if (rc == 0) {
        std::vector<long> pos_issue[2] = { std::vector<long>(n_layer, -1), std::vector<long>(n_layer, -1) };
        std::vector<long> pos_wait[2]  = { std::vector<long>(n_layer, -1), std::vector<long>(n_layer, -1) };
        size_t next_wait[2]  = { 0, 0 };
        size_t next_issue[2] = { 0, 0 };
        int open = 0, max_open = 0;
        for (size_t i = 0; i < g_log.size(); ++i) {
            const log_entry & e = g_log[i];
            if (e.kind == 'I') {
                if (e.layer != (int) next_issue[e.group]++) {
                    return fail("issues of a lane out of layer order");
                }
                pos_issue[e.group][e.layer] = (long) i;
                if (++open > max_open) {
                    max_open = open;
                }
            } else {
                if (e.layer != (int) next_wait[e.group]++) {
                    return fail("waits out of issue order (FIFO would mismatch)");
                }
                if (pos_issue[e.group][e.layer] < 0 || pos_issue[e.group][e.layer] > (long) i) {
                    return fail("wait before its issue");
                }
                pos_wait[e.group][e.layer] = (long) i;
                --open;
            }
        }
        for (int l = 0; l + 1 < n_layer; ++l) {
            // issue(A,l+1) is in flight while wait(B,l) is pending
            if (!(pos_issue[0][l + 1] < pos_wait[1][l])) {
                return fail("issue(A,l+1) must precede wait(B,l)");
            }
        }
        if (open != 0) {
            return fail("dispatches left open at end");
        }
        if (max_open != 2) {
            std::printf("max open %d\n", max_open);
            return fail("max open dispatches must be 2");
        }
        std::printf("open dispatches: max %d (k_max_open_dispatches = 2)\n", max_open);
    }

    ggml_free(ctx);
    if (rc == 0) {
        std::printf("T3 OK: %zu events, order A0,B0,WA0,IA1,WB0,IB1,... issue(A,l+1) before wait(B,l)\n", g_log.size());
    }
    return rc;
}
