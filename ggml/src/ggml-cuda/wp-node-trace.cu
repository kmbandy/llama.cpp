#include "wp-node-trace.cuh"
#include "common.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// See wp-node-trace.cuh for the contract. Ring size fixed at 64 per the
// WP_NODE_TRACE spec; events are created lazily per (device, ring slot) and
// never destroyed -- they are reused in place across wrap-arounds.
static constexpr int WP_NODE_TRACE_RING = 512;

struct wp_node_trace_entry {
    bool        valid      = false;
    int         node_idx   = -1;
    uint64_t    call_idx   = 0; // which ggml_backend_cuda_graph_compute() invocation (ubatch/graph counter)
    ggml_op     op         = GGML_OP_NONE;
    char        name[GGML_MAX_NAME] = {0};
    int64_t     ne[GGML_MAX_DIMS]      = {0, 0, 0, 0};
    int64_t     src0_ne[GGML_MAX_DIMS] = {0, 0, 0, 0};
    bool        has_src0   = false;
    bool        in_capture = false; // true => "graph-replay": no event was recorded for this launch
    cudaEvent_t event      = nullptr;
    cudaStream_t stream    = nullptr; // stream the node was launched on
    char        bufs[96]   = {0};     // buffer names of dst/src0/src1
};

struct wp_node_trace_device {
    wp_node_trace_entry   ring[WP_NODE_TRACE_RING];
    // Monotonically increasing count of nodes recorded for this device.
    // Single writer (the device's compute thread) -- relaxed is enough; the
    // watchdog thread reading this concurrently only uses it to pick a scan
    // order and tolerates a stale/torn value (best-effort diagnostics).
    std::atomic<uint32_t> head{0};
    // Bumped every time node_idx==0 is recorded, i.e. once per
    // ggml_backend_cuda_graph_compute() call for this device -- the
    // "ubatch/graph counter" the ring stores alongside each node.
    uint64_t               call_idx{0};
};

static wp_node_trace_device g_wp_node_trace[GGML_CUDA_MAX_DEVICES];

bool wp_node_trace_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("WP_NODE_TRACE");
        return env != nullptr && strcmp(env, "1") == 0;
    }();
    return enabled;
}

void wp_node_trace_record(int device, cudaStream_t stream, const ggml_tensor * node, int node_idx, bool in_capture) {
    if (!wp_node_trace_enabled()) {
        return;
    }
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || node == nullptr) {
        return;
    }

    wp_node_trace_device & dev = g_wp_node_trace[device];
    const uint32_t         count = dev.head.load(std::memory_order_relaxed);
    const int               slot  = (int) (count % WP_NODE_TRACE_RING);
    wp_node_trace_entry &   e     = dev.ring[slot];

    if (node_idx == 0) {
        dev.call_idx++;
    }

    e.node_idx   = node_idx;
    e.call_idx   = dev.call_idx;
    e.op         = node->op;
    snprintf(e.name, sizeof(e.name), "%s", node->name);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        e.ne[d] = node->ne[d];
    }
    if (node->src[0] != nullptr) {
        e.has_src0 = true;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            e.src0_ne[d] = node->src[0]->ne[d];
        }
    } else {
        e.has_src0 = false;
    }
    e.in_capture = in_capture;
    e.stream     = stream;
    {
        const char * b0 = node->buffer ? ggml_backend_buffer_name(node->buffer) : "-";
        const char * b1 = node->src[0] && node->src[0]->buffer ? ggml_backend_buffer_name(node->src[0]->buffer) : "-";
        const char * b2 = node->src[1] && node->src[1]->buffer ? ggml_backend_buffer_name(node->src[1]->buffer) : "-";
        snprintf(e.bufs, sizeof(e.bufs), "dst=%s s0=%s s1=%s", b0, b1, b2);
    }

    if (!in_capture) {
        // Recording an event on a stream that is being captured would itself
        // become part of the captured graph and fire on every future replay
        // (not what we want -- see the header comment), so events are only
        // ever created/recorded for non-captured (eager) launches.
        if (e.event == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&e.event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(e.event, stream));
    }
    // e.event is left as-is (either null, or a stale event from a previous
    // occupant of this slot) when in_capture -- the dump below treats
    // in_capture entries as "graph-replay" and never queries e.event for them.

    e.valid = true;
    dev.head.store(count + 1, std::memory_order_relaxed);
}

// cudaEventQuery status as a short label. Never CUDA_CHECK -- this runs on
// the watchdog thread and must never abort the process while diagnosing a
// stall.
static const char * wp_node_trace_event_status(cudaEvent_t ev) {
    if (ev == nullptr) {
        return "n/a";
    }
    const cudaError_t st = cudaEventQuery(ev);
    if (st == cudaSuccess) {
        return "ready";
    }
    if (st == cudaErrorNotReady) {
        return "not-ready";
    }
    return "err";
}

static void wp_node_trace_format_entry(char * buf, size_t n, const wp_node_trace_entry & e) {
    if (!e.valid) {
        snprintf(buf, n, "n/a");
        return;
    }
    if (e.in_capture) {
        snprintf(buf, n, "%d:%s:%s:graph-replay", e.node_idx, ggml_op_name(e.op), e.name);
        return;
    }
    snprintf(buf, n, "%d:%s:%s", e.node_idx, ggml_op_name(e.op), e.name);
}

void wp_node_trace_dump(int device) {
    if (!wp_node_trace_enabled()) {
        return;
    }
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }

    wp_node_trace_device & dev   = g_wp_node_trace[device];
    const uint32_t          count = dev.head.load(std::memory_order_relaxed);
    if (count == 0) {
        fprintf(stderr, "wp ar-watchdog: node-trace dev=%d ring empty\n", device);
        return;
    }

    const uint32_t n_entries = count < (uint32_t) WP_NODE_TRACE_RING ? count : (uint32_t) WP_NODE_TRACE_RING;

    // Walk newest -> oldest. "last_launched" is simply the newest entry.
    // "last_completed" is the newest entry whose event (if any) is ready, or
    // whose in_capture flag marks it graph-replay (a replayed graph launch
    // has no per-node event; we cannot tell node-level completion inside it,
    // so it is reported separately as the replay region rather than as
    // completed/stuck). "stuck_after" is the first (in launch order, i.e.
    // walking from oldest-of-the-not-ready-run) not-ready node after
    // last_completed -- the node that is running/stuck.
    const wp_node_trace_entry * last_launched  = nullptr;
    const wp_node_trace_entry * last_completed = nullptr;
    const wp_node_trace_entry * stuck_after    = nullptr;
    bool                        replay_region  = false;

    for (uint32_t back = 0; back < n_entries; ++back) {
        const uint32_t idx = (count - 1 - back) % (uint32_t) WP_NODE_TRACE_RING;
        const wp_node_trace_entry & e = dev.ring[idx]; // best-effort: may race with a concurrent record()
        if (!e.valid) {
            continue;
        }
        if (last_launched == nullptr) {
            last_launched = &e;
        }
        if (e.in_capture) {
            // Can't query per-node completion inside a captured/replayed
            // graph -- treat the whole run of such entries as the possible
            // stuck region rather than as completed.
            stuck_after   = &e;
            replay_region = true;
            continue;
        }
        const char * status = wp_node_trace_event_status(e.event);
        if (strcmp(status, "ready") == 0) {
            last_completed = &e;
            break;
        }
        // not-ready (or err/n/a): this node hadn't finished as of last
        // launch -- it is (so far) the oldest still-outstanding node seen
        // walking backwards, i.e. the current candidate for "stuck".
        stuck_after   = &e;
        replay_region = false;
    }

    char launched_buf[160];
    char completed_buf[160];
    char stuck_buf[192];
    wp_node_trace_format_entry(launched_buf, sizeof(launched_buf), last_launched ? *last_launched : wp_node_trace_entry{});
    wp_node_trace_format_entry(completed_buf, sizeof(completed_buf), last_completed ? *last_completed : wp_node_trace_entry{});

    if (stuck_after != nullptr) {
        char base[160];
        wp_node_trace_format_entry(base, sizeof(base), *stuck_after);
        const wp_node_trace_entry & s = *stuck_after;
        snprintf(stuck_buf, sizeof(stuck_buf),
                 "%s ne=[%lld,%lld,%lld,%lld] src0=[%lld,%lld,%lld,%lld]%s",
                 base,
                 (long long) s.ne[0], (long long) s.ne[1], (long long) s.ne[2], (long long) s.ne[3],
                 (long long) (s.has_src0 ? s.src0_ne[0] : 0),
                 (long long) (s.has_src0 ? s.src0_ne[1] : 0),
                 (long long) (s.has_src0 ? s.src0_ne[2] : 0),
                 (long long) (s.has_src0 ? s.src0_ne[3] : 0),
                 replay_region ? " (replayed graph)" : "");
    } else {
        snprintf(stuck_buf, sizeof(stuck_buf), "none");
    }

    fprintf(stderr,
            "wp ar-watchdog: node-trace dev=%d last_launched=%s last_completed=%s stuck_after=%s\n",
            device, launched_buf, completed_buf, stuck_buf);

    // Full ring, oldest first, so the boundary between completed and
    // never-started nodes is visible (a stream blocked on an event shows as
    // a run of not-ready entries starting right after the last ready one).
    for (uint32_t k = 0; k < n_entries; ++k) {
        const uint32_t idx = (count - n_entries + k) % (uint32_t) WP_NODE_TRACE_RING;
        const wp_node_trace_entry & e = dev.ring[idx];
        if (!e.valid) {
            continue;
        }
        char buf[160];
        wp_node_trace_format_entry(buf, sizeof(buf), e);
        fprintf(stderr, "wp ar-watchdog:   node dev=%d call=%llu stream=%p %s ne=[%lld,%lld] %s -> %s\n",
                device, (unsigned long long) e.call_idx, (void *) e.stream, buf,
                (long long) e.ne[0], (long long) e.ne[1], e.bufs,
                e.in_capture ? "replay" : wp_node_trace_event_status(e.event));
    }
    fflush(stderr);
}
