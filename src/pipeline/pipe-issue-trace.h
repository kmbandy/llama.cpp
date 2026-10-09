// WP_ISSUE_TRACE=1 -- per-call timers for the spine's expert-dispatch ISSUE path
// (graph_dispatcher::compute_issue and the dispatcher::begin_dispatch it calls).
//
// Default OFF: every hook reduces to one thread_local pointer test. When ON,
// each compute_issue call fills one record on the caller thread (no I/O, no
// locks on that path); finished records go into a preallocated ring that is
// written to WP_ISSUE_TRACE_FILE (default /tmp/wp-issue-trace.<pid>.tsv) every
// WP_ISSUE_TRACE_FLUSH calls (default 256) and at process exit. The periodic
// flush is a single fwrite of at most FLUSH records and is the only I/O.
//
// Phases are wall-clock on the issuing thread. Several are NESTED (the BD_*
// phases contain PR_* / IR_* ; HINT_* and ACT_* / ASSIGN_* sit inside
// t_total_us). Pool-thread encode work is charged to its caller-side wall time.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <unistd.h>

namespace wp_issue_trace {

#define WP_IT_PHASES(X)                                  \
    X(WAIT_DRAIN,        "wait_drain")                   \
    X(IO_LOCK,           "io_lock")                      \
    X(ACT_COPY,          "act_copy")                     \
    X(ASSIGN_CHUNK,      "assign_chunk")                 \
    X(FULL_ACT_COPY,     "full_act_copy")                \
    X(ASSIGN_FULL,       "assign_full")                  \
    X(HINT_PRE_FLUSH,    "hint_pre_flush")               \
    X(HINT_PRE_PREFETCH, "hint_pre_prefetch")            \
    X(BEGIN_DISPATCH,    "begin_dispatch")               \
    X(POST_SEND_HINTS,   "post_send_hints")              \
    X(POST_ENQUEUE_PRED, "post_enqueue_pred")            \
    X(POST_FLUSH_HINTS,  "post_flush_hints")             \
    X(POST_PREFETCH,     "post_prefetch")                \
    X(BD_LOCALITY_DUMP,  "bd_locality_dump")             \
    X(BD_SPLIT_DEFER,    "bd_split_defer")               \
    X(BD_PLAN_IMM,       "bd_plan_imm")                  \
    X(BD_PLAN_DEF,       "bd_plan_def")                  \
    X(PR_CLASSIFY,       "pr_classify")                  \
    X(PR_PACK,           "pr_pack")                      \
    X(PR_ENCODE,         "pr_encode")                    \
    X(BD_DEDUP,          "bd_dedup")                     \
    X(BD_ISSUE_IMM,      "bd_issue_imm")                 \
    X(BD_RECV_AHEAD,     "bd_recv_ahead")                \
    X(BD_ISSUE_DEF,      "bd_issue_def")                 \
    X(BD_COLLECT_DEFER,  "bd_collect_deferred")          \
    X(IR_SEND_BLOCK,     "ir_send_blocking")             \
    X(IR_ENQUEUE,        "ir_enqueue")                   \
    X(IR_ENQ_LOCK,       "ir_enqueue_lock_wait")         \
    X(IR_FLUSH_WRITER,   "ir_flush_writer_wait")

enum phase_id : int {
#define WP_IT_ENUM(id, name) PH_##id,
    WP_IT_PHASES(WP_IT_ENUM)
#undef WP_IT_ENUM
    PH_COUNT
};

static const char * const k_phase_names[PH_COUNT] = {
#define WP_IT_NAME(id, name) name,
    WP_IT_PHASES(WP_IT_NAME)
#undef WP_IT_NAME
};

constexpr int      MAX_WORKERS  = 16;   // bytes_w0..bytes_w15 columns (unused = 0)
constexpr size_t   RING_MAX     = 4096;

struct record {
    uint64_t seq         = 0;
    int32_t  layer       = -1;
    uint32_t n_tokens    = 0;
    uint32_t chunk_index = 0;
    uint32_t n_workers   = 0;
    uint64_t t_total_ns  = 0;
    uint64_t ph_ns[PH_COUNT] = {};
    uint64_t bytes[MAX_WORKERS] = {};
    uint64_t n_hint_frames = 0;
    uint64_t hint_bytes    = 0;
};

struct state {
    bool        enabled = false;
    size_t      flush_every = 256;
    std::string path;
    std::FILE * file = nullptr;
    bool        header_written = false;
    uint64_t    seq = 0;
    std::vector<record> ring;
    std::mutex  mu;
    bool        atexit_registered = false;
};

inline state & st() {
    static state s;
    static const bool init = [] {
        const char * e = std::getenv("WP_ISSUE_TRACE");
        s.enabled = e != nullptr && e[0] == '1';
        if (!s.enabled) {
            return true;
        }
        const char * f = std::getenv("WP_ISSUE_TRACE_FILE");
        if (f != nullptr && f[0] != '\0') {
            s.path = f;
        } else {
            s.path = "/tmp/wp-issue-trace." + std::to_string((long) getpid()) + ".tsv";
        }
        const char * fl = std::getenv("WP_ISSUE_TRACE_FLUSH");
        if (fl != nullptr && fl[0] != '\0') {
            const long v = std::strtol(fl, nullptr, 10);
            if (v > 0) {
                s.flush_every = (size_t) (v > (long) RING_MAX ? RING_MAX : v);
            }
        }
        s.ring.reserve(RING_MAX);
        return true;
    }();
    (void) init;
    return s;
}

// Current record for this thread; null when tracing is off or outside a
// compute_issue call (e.g. llama_context::decode()'s flush_predicted_hints).
inline thread_local record * cur = nullptr;

inline uint64_t mark() {
    if (cur == nullptr) {
        return 0;
    }
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void add(int id, uint64_t t0) {
    if (cur == nullptr || t0 == 0) {
        return;
    }
    const uint64_t t1 = mark();
    cur->ph_ns[id] += t1 - t0;
}

inline void add_bytes(uint32_t worker, uint64_t n) {
    if (cur != nullptr && worker < MAX_WORKERS) {
        cur->bytes[worker] += n;
    }
}

inline void note_workers(uint32_t n) {
    if (cur != nullptr && n > cur->n_workers) {
        cur->n_workers = n;
    }
}

inline void note_hint_frame(uint64_t bytes) {
    if (cur != nullptr) {
        ++cur->n_hint_frames;
        cur->hint_bytes += bytes;
    }
}

inline void flush_locked(state & s) {
    if (s.ring.empty()) {
        return;
    }
    if (s.file == nullptr) {
        s.file = std::fopen(s.path.c_str(), "w");
        if (s.file == nullptr) {
            s.ring.clear();
            return;
        }
    }
    if (!s.header_written) {
        std::fprintf(s.file, "seq\tlayer\tn_tokens\tchunk_index\tn_workers\tt_total_us");
        for (int i = 0; i < PH_COUNT; ++i) {
            std::fprintf(s.file, "\tph_%s_us", k_phase_names[i]);
        }
        for (int w = 0; w < MAX_WORKERS; ++w) {
            std::fprintf(s.file, "\tbytes_w%d", w);
        }
        std::fprintf(s.file, "\tn_hint_frames\thint_bytes\n");
        s.header_written = true;
    }
    for (const record & r : s.ring) {
        std::fprintf(s.file, "%llu\t%d\t%u\t%u\t%u\t%.3f",
                     (unsigned long long) r.seq, r.layer, r.n_tokens, r.chunk_index, r.n_workers,
                     (double) r.t_total_ns / 1e3);
        for (int i = 0; i < PH_COUNT; ++i) {
            std::fprintf(s.file, "\t%.3f", (double) r.ph_ns[i] / 1e3);
        }
        for (int w = 0; w < MAX_WORKERS; ++w) {
            std::fprintf(s.file, "\t%llu", (unsigned long long) r.bytes[w]);
        }
        std::fprintf(s.file, "\t%llu\t%llu\n",
                     (unsigned long long) r.n_hint_frames, (unsigned long long) r.hint_bytes);
    }
    std::fflush(s.file);
    s.ring.clear();
}

inline void flush_now() {
    state & s = st();
    if (!s.enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mu);
    flush_locked(s);
}

inline void at_exit_flush() {
    flush_now();
    state & s = st();
    if (s.file != nullptr) {
        std::fclose(s.file);
        s.file = nullptr;
    }
}

// RAII per compute_issue call. Constructed with the owner's layer/chunk; the
// record is committed (ring push, periodic flush) on every exit path.
struct call_guard {
    record rec;
    bool   active = false;
    uint64_t t0 = 0;

    call_guard(int32_t layer, uint32_t n_tokens, uint32_t chunk_index) {
        state & s = st();
        if (!s.enabled) {
            return;
        }
        active = true;
        rec.layer       = layer;
        rec.n_tokens    = n_tokens;
        rec.chunk_index = chunk_index;
        t0 = mark_abs();
        cur = &rec;
        if (!s.atexit_registered) {
            s.atexit_registered = true;
            std::atexit(at_exit_flush);
        }
    }

    ~call_guard() {
        if (!active) {
            return;
        }
        rec.t_total_ns = mark_abs() - t0;
        cur = nullptr;
        state & s = st();
        std::lock_guard<std::mutex> lock(s.mu);
        rec.seq = ++s.seq;
        s.ring.push_back(rec);
        if (s.ring.size() >= s.flush_every) {
            flush_locked(s);
        }
    }

    static uint64_t mark_abs() {
        return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

}  // namespace wp_issue_trace
