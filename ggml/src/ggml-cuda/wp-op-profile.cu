#include "wp-op-profile.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

struct pending_pair {
    cudaEvent_t start;
    cudaEvent_t end;
    std::string key;
    bool        free_start = true;
    bool        free_end   = true;
    bool        is_gap     = false;
};

struct acc {
    double ms = 0.0;
    uint64_t n = 0;
};

struct dev_state {
    std::vector<cudaEvent_t>   free_events;
    std::deque<pending_pair>   pending;
    std::map<std::string, acc> totals;      // key -> accumulated
    double                     graph_wall_ms = 0.0; // host wall spent inside graph_compute (approx.)
    uint64_t                   n_graphs = 0;
    std::string                bucket = "?";
    cudaEvent_t                cur_start = nullptr;
    // End event of the previous node on the compute stream, kept alive so the
    // idle gap before the next node can be measured (GAP rows). Owned by the
    // gap pair that consumes it, or freed here when a graph ends without a
    // successor.
    cudaEvent_t                prev_end = nullptr;
    cudaEvent_t                span_start = nullptr;
    uint64_t                   prev_graph = 0;
    double                     gap_ms_sum = 0.0;
    std::chrono::steady_clock::time_point last_print = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point graph_t0;
    bool                       in_graph = false;
};

dev_state g_dev[GGML_CUDA_MAX_DEVICES];

double print_interval_s() {
    static const double s = [] {
        const char * e = getenv("WP_OP_PROFILE_S");
        return e != nullptr && atof(e) > 0 ? atof(e) : 5.0;
    }();
    return s;
}

// Row cap for the per-label table; 0 = unlimited.
int print_rows() {
    static const int n = [] {
        const char * e = getenv("WP_OP_PROFILE_ROWS");
        return e != nullptr && atoi(e) >= 0 ? atoi(e) : 40;
    }();
    return n;
}

// Split a per-label key ("<bucket> <op-name-with-fusion-suffix> <node-label>",
// or a GAP/GAPg-prefixed variant, or a bucket-only span key with no node
// label such as "<bucket> AR_END_WAIT") into the (class, op-group) pair used
// by the by-op aggregated table. GAP/GAPg rows fold into a single "GAP" /
// "GAPg" group per class regardless of the op that follows.
std::pair<std::string, std::string> by_op_group(const std::string & key) {
    const size_t sp = key.find(' ');
    if (sp == std::string::npos) {
        return { key, "" };
    }
    const std::string cls  = key.substr(0, sp);
    const std::string rest = key.substr(sp + 1);
    if (rest.rfind("GAPg>", 0) == 0) {
        return { cls, "GAPg" };
    }
    if (rest.rfind("GAP>", 0) == 0) {
        return { cls, "GAP" };
    }
    const size_t last_sp = rest.rfind(' ');
    if (last_sp == std::string::npos) {
        // no trailing per-node label (e.g. AR_END_WAIT, AR_END_UNPACK, GRAPH_REPLAY)
        return { cls, rest };
    }
    return { cls, rest.substr(0, last_sp) };
}

cudaEvent_t take_event(dev_state & d) {
    if (!d.free_events.empty()) {
        cudaEvent_t e = d.free_events.back();
        d.free_events.pop_back();
        return e;
    }
    cudaEvent_t e = nullptr;
    CUDA_CHECK(cudaEventCreate(&e)); // timing enabled
    return e;
}

// Strip the trailing "-<layer>" so per-layer tensors fold into one key.
std::string strip_layer(const char * name) {
    std::string s(name);
    size_t p = s.rfind('-');
    if (p != std::string::npos && p + 1 < s.size()) {
        bool digits = true;
        for (size_t i = p + 1; i < s.size(); ++i) {
            if (s[i] < '0' || s[i] > '9') { digits = false; break; }
        }
        if (digits) {
            s.erase(p);
        }
    }
    return s;
}

std::string make_key(const ggml_tensor * node, int n_fused) {
    char buf[256];
    const char * op = node->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(node)) : ggml_op_name(node->op);
    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        snprintf(buf, sizeof(buf), "%s(%s)%s %s", op,
                 node->src[0] ? ggml_type_name(node->src[0]->type) : "?",
                 n_fused > 1 ? "+f" : "", strip_layer(node->name).c_str());
    } else {
        snprintf(buf, sizeof(buf), "%s%s %s", op, n_fused > 1 ? "+f" : "", strip_layer(node->name).c_str());
    }
    return std::string(buf);
}

void drain(dev_state & d, bool all) {
    while (!d.pending.empty()) {
        pending_pair & p = d.pending.front();
        if (!all) {
            const cudaError_t st = cudaEventQuery(p.end);
            if (st == cudaErrorNotReady) {
                break;
            }
        } else {
            cudaEventSynchronize(p.end);
        }
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, p.start, p.end) == cudaSuccess) {
            acc & a = d.totals[p.key];
            a.ms += ms;
            a.n  += 1;
            if (p.is_gap) {
                d.gap_ms_sum += ms;
            }
        } else {
            (void) cudaGetLastError();
        }
        if (p.free_start) {
            d.free_events.push_back(p.start);
        }
        if (p.free_end) {
            d.free_events.push_back(p.end);
        }
        d.pending.pop_front();
    }
}

void maybe_print(int device, dev_state & d) {
    const auto now = std::chrono::steady_clock::now();
    const double since = std::chrono::duration<double>(now - d.last_print).count();
    if (since < print_interval_s()) {
        return;
    }
    d.last_print = now;
    if (d.totals.empty()) {
        return;
    }
    double total = 0.0;
    std::vector<std::pair<std::string, acc>> rows(d.totals.begin(), d.totals.end());
    for (const auto & r : rows) {
        total += r.second.ms;
    }
    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) { return a.second.ms > b.second.ms; });
    // gpu_op_sum counts ops only; GAP rows (idle on the compute stream between
    // consecutive ops, keyed by the op that follows, GAPg when the gap spans a
    // graph_compute boundary) are reported separately and excluded from total.
    total -= d.gap_ms_sum;
    fprintf(stderr, "wp op-profile dev=%d window=%.1fs graphs=%llu graph_wall=%.0fms gpu_op_sum=%.0fms gap_sum=%.0fms\n",
            device, since, (unsigned long long) d.n_graphs, d.graph_wall_ms, total, d.gap_ms_sum);
    const int cap = print_rows();
    int shown = 0;
    for (const auto & r : rows) {
        if (cap > 0 && shown++ >= cap) {
            break;
        }
        fprintf(stderr, "wp op-profile dev=%d   %7.1fms %5.1f%% n=%-6llu %s\n",
                device, r.second.ms, total > 0.0 ? 100.0 * r.second.ms / total : 0.0, (unsigned long long) r.second.n, r.first.c_str());
    }

    // Aggregated by-op table: fold the per-node label out of each key so
    // identical ops in different layers collapse into one group. GAP/GAPg
    // rows fold into a single "GAP"/"GAPg" group per class.
    std::map<std::pair<std::string, std::string>, acc> by_op;
    double wait_ms = 0.0, unpack_ms = 0.0;
    for (const auto & r : rows) {
        const std::pair<std::string, std::string> g = by_op_group(r.first);
        acc & a = by_op[g];
        a.ms += r.second.ms;
        a.n  += r.second.n;
        if (g.second == "AR_END_WAIT") {
            wait_ms += r.second.ms;
        } else if (g.second == "AR_END_UNPACK") {
            unpack_ms += r.second.ms;
        }
    }
    std::vector<std::pair<std::pair<std::string, std::string>, acc>> by_op_rows(by_op.begin(), by_op.end());
    std::sort(by_op_rows.begin(), by_op_rows.end(),
              [](const auto & a, const auto & b) { return a.second.ms > b.second.ms; });
    fprintf(stderr, "wp op-profile dev=%d by-op:\n", device);
    for (const auto & r : by_op_rows) {
        fprintf(stderr, "wp op-profile dev=%d   %7.1fms %5.1f%% n=%-6llu %s %s\n",
                device, r.second.ms, d.graph_wall_ms > 0.0 ? 100.0 * r.second.ms / d.graph_wall_ms : 0.0,
                (unsigned long long) r.second.n, r.first.first.c_str(), r.first.second.c_str());
    }

    const double other_ms = total - wait_ms - unpack_ms;
    fprintf(stderr, "wp op-profile dev=%d totals: wall=%.0fms ops=%.0fms gap=%.0fms wait=%.0fms unpack=%.0fms other=%.0fms\n",
            device, d.graph_wall_ms, total, d.gap_ms_sum, wait_ms, unpack_ms, other_ms);
    fflush(stderr);
    d.totals.clear();
    d.graph_wall_ms = 0.0;
    d.gap_ms_sum = 0.0;
    d.n_graphs = 0;
}

} // namespace

bool wp_op_profile_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("WP_OP_PROFILE");
        return e != nullptr && strcmp(e, "1") == 0;
    }();
    return enabled;
}

void wp_op_profile_begin_graph(int device, const ggml_cgraph * cgraph) {
    if (!wp_op_profile_enabled() || device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    dev_state & d = g_dev[device];
    const auto now = std::chrono::steady_clock::now();
    if (d.in_graph) {
        d.graph_wall_ms += std::chrono::duration<double, std::milli>(now - d.graph_t0).count();
    }
    d.graph_t0 = now;
    d.in_graph = true;
    d.n_graphs++;
    // token bucket = ne[1] of the first MUL_MAT in the graph
    int64_t n_tok = -1;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) {
            n_tok = t->ne[1];
            break;
        }
    }
    char b[32];
    if (n_tok < 0) {
        snprintf(b, sizeof(b), "n?");
    } else if (n_tok <= 16) {
        snprintf(b, sizeof(b), "tg%lld", (long long) n_tok);
    } else {
        snprintf(b, sizeof(b), "pp%lld", (long long) n_tok);
    }
    d.bucket = b;
    drain(d, false);
    maybe_print(device, d);
}

void wp_op_profile_begin_node(int device, cudaStream_t stream, bool in_capture) {
    if (!wp_op_profile_enabled() || in_capture || device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    dev_state & d = g_dev[device];
    d.cur_start = take_event(d);
    CUDA_CHECK(cudaEventRecord(d.cur_start, stream));
}

void wp_op_profile_end_node(int device, cudaStream_t stream, const ggml_tensor * node, int n_fused, bool in_capture) {
    if (!wp_op_profile_enabled() || in_capture || device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    dev_state & d = g_dev[device];
    if (d.cur_start == nullptr) {
        return;
    }
    cudaEvent_t end = take_event(d);
    CUDA_CHECK(cudaEventRecord(end, stream));
    const std::string key = make_key(node, n_fused);
    if (d.prev_end != nullptr) {
        // idle between the previous node's end and this node's start; the gap
        // pair owns prev_end, the op pair below owns cur_start
        const bool cross_graph = d.prev_graph != d.n_graphs;
        pending_pair g;
        g.start = d.prev_end; g.end = d.cur_start; g.is_gap = true;
        g.free_start = true; g.free_end = false;
        g.key = d.bucket + (cross_graph ? " GAPg>" : " GAP>") + key;
        d.pending.push_back(g);
    }
    pending_pair op;
    op.start = d.cur_start; op.end = end; op.free_start = true; op.free_end = false;
    op.key = d.bucket + " " + key;
    d.pending.push_back(op);
    d.prev_end   = end;
    d.prev_graph = d.n_graphs;
    d.cur_start  = nullptr;
}

void wp_op_profile_begin_replay(int device, cudaStream_t stream) {
    wp_op_profile_begin_node(device, stream, false);
}

void wp_op_profile_end_replay(int device, cudaStream_t stream) {
    if (!wp_op_profile_enabled() || device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    dev_state & d = g_dev[device];
    if (d.cur_start == nullptr) {
        return;
    }
    cudaEvent_t end = take_event(d);
    CUDA_CHECK(cudaEventRecord(end, stream));
    if (d.prev_end != nullptr) {
        pending_pair g;
        g.start = d.prev_end; g.end = d.cur_start; g.is_gap = true;
        g.free_start = true; g.free_end = false;
        g.key = d.bucket + (d.prev_graph != d.n_graphs ? " GAPg>" : " GAP>") + "GRAPH_REPLAY";
        d.pending.push_back(g);
    }
    pending_pair op;
    op.start = d.cur_start; op.end = end; op.free_start = true; op.free_end = false;
    op.key = d.bucket + " GRAPH_REPLAY";
    d.pending.push_back(op);
    d.prev_end   = end;
    d.prev_graph = d.n_graphs;
    d.cur_start  = nullptr;
}

void wp_op_profile_span_begin(int device, cudaStream_t stream) {
    if (!wp_op_profile_enabled() || device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    dev_state & d = g_dev[device];
    cudaEvent_t e = take_event(d);
    CUDA_CHECK(cudaEventRecord(e, stream));
    d.span_start = e;
}

void wp_op_profile_span_end(int device, cudaStream_t stream, const char * key) {
    if (!wp_op_profile_enabled() || device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    dev_state & d = g_dev[device];
    if (d.span_start == nullptr) {
        return;
    }
    cudaEvent_t end = take_event(d);
    CUDA_CHECK(cudaEventRecord(end, stream));
    pending_pair op;
    op.start = d.span_start; op.end = end; op.free_start = true; op.free_end = true;
    op.key = d.bucket + " " + key;
    d.pending.push_back(op);
    d.span_start = nullptr;
}
