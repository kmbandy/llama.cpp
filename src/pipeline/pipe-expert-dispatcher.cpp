#include "pipe-expert-dispatcher.h"

#include "ggml.h"
#include "llama-impl.h"
#include "pipe-transport.h"
#include "pipe-reduce-simd.h"

#include <algorithm>
#include <numeric>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// poll() for concurrent harvest of worker responses (POSIX; the dispatcher
// already assumes POSIX sockets).
#include <poll.h>
#include <cerrno>

namespace pipe_expert_dispatcher {
namespace {

inproc_backend_factory g_inproc_factory = nullptr;

static bool dispatch_hash_trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("WP_DISPATCH_HASH_TRACE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

static uint64_t dispatch_hash_fnv1a(const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

std::atomic<uint64_t> g_dispatch_hash_seq{0};

using dispatch_clock = std::chrono::steady_clock;

bool dispatch_stats_enabled() {
    const char * value = std::getenv("WP_DISPATCH_STATS");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool speed_split_enabled() {
    const char * value = std::getenv("WP_DISPATCH_SPEED_SPLIT");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

// WP_DISPATCH_STATIC_ASSIGN -- default ON. See choose_worker for the full
// reasoning; hoisted out of that function's local static so the prefetch-hint
// path can ask the same question, because a hint is only safe to send when the
// worker choice is predictable. ONE definition, so the two can never disagree.
bool static_assign_enabled() {
    const char * value = std::getenv("WP_DISPATCH_STATIC_ASSIGN");
    return value == nullptr || value[0] != '0';
}

// WP_HINT_INFLIGHT -- default OFF. The worker dispatch loop consumes complete
// frames and type-dispatches hints, so a hint can safely follow an outstanding
// request on the same socket. Keep the old guard unless this is enabled.
bool hint_inflight_enabled() {
    const char * value = std::getenv("WP_HINT_INFLIGHT");
    return value != nullptr && value[0] == '1';
}

// Decode/verify (n_tokens <= max) prefer this worker port among candidates.
// DEFAULT OFF. Opt in with WP_DISPATCH_DECODE_PORT=8803 (GTX 1070). This
// shifts 0-84 decode off the RX 480; it does not make either card faster.
int decode_prefer_port_enabled() {
    const char * value = std::getenv("WP_DISPATCH_DECODE_PORT");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    return std::atoi(value);
}

uint32_t decode_max_tokens_enabled() {
    const char * value = std::getenv("WP_DISPATCH_DECODE_MAX_TOKENS");
    if (value == nullptr || value[0] == '\0') {
        return 8;
    }
    const int n = std::atoi(value);
    return n < 0 ? 0u : (uint32_t) n;
}

// Default off so existing harness flows preserve their fail-fast startup.
int dispatch_connect_retry_seconds() {
    const char * value = std::getenv("WP_DISPATCH_CONNECT_RETRY_S");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    const long seconds = std::strtol(value, nullptr, 10);
    return seconds > 0 ? (int) std::min(seconds, (long) INT_MAX) : 0;
}

// WP_ASYNC_ISSUE -- default OFF as of 2026-08-08 round 8. Decode measured a
// +60 ms/dispatch net regression and an unexplained multi-second stall mode;
// see docs/dev/2026-08-08-runs.txt. Opt in with WP_ASYNC_ISSUE=1 for the
// prefill retest at code2000 length.
bool async_issue_enabled() {
    const char * value = std::getenv("WP_ASYNC_ISSUE");
    return value != nullptr && value[0] != '0';
}

bool split_frame_enabled() {
    const char * value = std::getenv("WP_SPLIT_FRAME");
    return value != nullptr && value[0] != '0';
}

// WP_CONCURRENT_ISSUE -- default OFF. Independent of, and deliberately NOT
// built on, WP_ASYNC_ISSUE's socket_writer (a FIFO queue drained by a thread
// woken per-frame via condvar, MAX_QUEUE=8 backpressure, payload moved into
// the queued frame). That shape was measured net-negative (+60 ms/dispatch,
// occasional multi-second stalls -- see async_issue_enabled() above and KG
// 777a57ff) and issue_requests() never waits for it, so a slow/wedged writer
// is only discovered lazily at the next await/enqueue.
//
// This flag instead uses a persistent one-job-slot sender per socket (see
// concurrent_sender): issue_requests() posts every worker's frame(s) without
// blocking, THEN joins every posted job before returning -- so the send wall
// time collapses from the SUM of per-link sends to the MAX, but the caller's
// contract (issue_requests returns only once every worker has been sent, and
// throws exactly as the serial path does on a failed send) is unchanged. No
// queue and nothing to grow unbounded: at most one job is ever outstanding
// per socket, because this thread posts then joins before that socket's next
// job is ever posted. No payload copy: the job holds a pointer straight into
// the request's payload/begin_payload/acts_payload vectors, which outlive the
// join.
bool concurrent_issue_enabled() {
    const char * value = std::getenv("WP_CONCURRENT_ISSUE");
    return value != nullptr && value[0] != '0';
}

// WP_DISPATCH_HARVEST=1 opts in to as-ready harvesting (poll() across all
// outstanding worker sockets, accumulate each partial the moment it arrives,
// then reduce in FIXED request order so the sum stays float-order
// deterministic regardless of arrival timing). Applies to both the harvest
// of a layer's immediate requests (see harvest_partials) and the fold of the
// previous layer's deferred requests (see collect_pending_deferred) -- both
// are the identical "await N sockets in a fixed worker order" shape.
//
// DEFAULT OFF, AND THAT IS A MEASURED DECISION, not caution. Measured
// 2026-08-02, load-matched back-to-back: 4.197 (off) vs 4.231 (on) tok/s,
// i.e. +0.8%, inside noise. The mechanism DOES work -- summed blocked time
// falls 152.72 -> 11.85 ms/token, every recv finds its data already waiting
// -- but the time simply moves into the poll wait, because the workers were
// ALREADY overlapping. Sum over layers of the MAX worker service was 74.97
// ms/token against a 155.8 ms dispatch wall, so ~81 ms/token was overhead
// that is NOT await ordering and this did not recover it. Keep the code (it
// is the instrument that measured wire latency directly: with harvest on,
// before_await minus worker service gives ~0.57-0.65 ms/request on the
// remote link and ~20 us on the R9700 loopback) but do not pay its
// complexity by default until a re-measurement under the current dense-spine
// / Tailscale-hop topology (a different regime than the 2026-08-02 test)
// justifies flipping it.
bool harvest_enabled() {
    const char * value = std::getenv("WP_DISPATCH_HARVEST");
    return value != nullptr && value[0] == '1';
}

// WP_UNPACK_OVERLAP=1 -- DEFAULT OFF. A second, independent opt-in into the
// SAME poll_harvest_receive() mechanism as WP_DISPATCH_HARVEST above, kept as
// its own flag rather than folded into that one because the two were measured
// for different things and the record for WP_DISPATCH_HARVEST (see the block
// above: net neutral for WAIT time, because the workers were already
// overlapping and the layer still completes at max(worker arrival) either
// way -- KG 777a57ff / "HARVEST-AS-READY ... CLOSED AS NEUTRAL") should not be
// read as a verdict on THIS question.
//
// The question this flag answers is narrower: today's fixed-order path
// (accumulate_partial, one worker at a time -- block-recv worker i, THEN
// wire-decode + fold worker i, THEN move to worker i+1) pays every worker's
// decode/deserialize cost serially, back to back, even for a worker whose
// bytes had already been sitting in its socket's receive buffer for
// milliseconds while we were still blocked on an earlier worker in fixed
// order. TCP reception itself is async in the kernel regardless of when we
// call recv() -- so if worker A (early in fixed order) is this layer's long
// pole and workers B..E answer sooner but later in fixed order, the naive
// loop still decodes A, then B, then C, ... entirely after A's slow arrival,
// serializing every worker's CPU decode cost onto the tail of the slowest
// worker's network wait. poll_harvest_receive() breaks that coupling: it
// decodes each response AS SOON AS ITS SOCKET IS READY (poll(), arrival
// order), so a fast worker's bytes get wire-decoded WHILE the slow worker's
// data is still in flight over the network -- genuine overlap of spine CPU
// decode work with the remaining network wait, using the SAME dispatch
// thread that already blocks in the harvest loop (zero new threads; compare
// WP_ASYNC_ISSUE, per-socket ISSUE-side writer threads, KG 777a57ff,
// measured NET-NEGATIVE with nondeterministic multi-second stalls -- this is
// deliberately not that shape). The final SUM/fold still walks `requests` in
// the same fixed worker-registration order as always (see the "WHY SUM IN
// FIXED ORDER" note on harvest_partials), so summation order and therefore
// the teacher-forced NLL are unaffected -- only WHEN each response gets
// wire-decoded moves, never in what order the decoded values are added.
//
// Because this reuses poll_harvest_receive()/harvest_partials() -- the exact
// same code as WP_DISPATCH_HARVEST=1 -- setting either flag takes the same
// branch in finish_dispatch()/collect_pending_deferred(). They are kept as
// two names because they answer two different measurement questions and a
// future re-test of one should not be read as also re-testing the other.
//
// Timer-attribution note: with this flag on, `stats.ns_wait` (issue -> last
// raw payload observed) now reflects the true last-TO-ARRIVE worker rather
// than always the fixed-order-last worker, and per-worker decode/fold cost
// that used to land after the last fixed-order recv (visible as "unpack" in
// the forward-budget WARN line) is now spread earlier, overlapped with wait
// on a slower peer. The WARN line's ns_unpack bucket can therefore shrink
// even though no work was removed -- see the note printed at construction
// below and the one added next to the forward-budget WARN format string.
bool unpack_overlap_enabled() {
    const char * value = std::getenv("WP_UNPACK_OVERLAP");
    return value != nullptr && value[0] == '1';
}

bool layer_trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("WP_DS4_LAYER_TRACE");
        return value != nullptr && value[0] != '\0';
    }();
    return enabled;
}

// WP_DISPATCH_UNION=1 -- measurement only, no behaviour change. Logs how many
// token rows a worker's assignments actually need versus how many it is sent.
// Read once at startup; a per-request getenv on the dispatch path would itself
// distort the thing being measured.
const bool s_union_stats = [] {
    const char * value = std::getenv("WP_DISPATCH_UNION");
    return value != nullptr && std::strcmp(value, "1") == 0;
}();

// WP_TEMPORAL_STATS=1 measures consecutive-token expert overlap. Read once so
// the disabled dispatch path only pays this boolean check.
const bool s_temporal_stats = [] {
    const char * value = std::getenv("WP_TEMPORAL_STATS");
    return value != nullptr && std::strcmp(value, "1") == 0;
}();

const char * s_routing_dump_path = [] {
    const char * value = std::getenv("WP_ROUTING_DUMP");
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}();

// WP_DISPATCH_GATHER=0 disables the spine-side activation gather. DEFAULT ON.
//
// Send a worker only the token rows that route to ITS experts, instead of the
// full [n_tokens x n_embd] tensor. NUMERICALLY EXACT: a dropped row has a zero
// routing weight for every expert that worker owns, so it would contribute
// exactly 0.0f to that worker's partial. This is what makes it safe where the
// f16-partial shrink was not -- that one was reverted (071f31b92, 9d9e5e4cc)
// for putting ~5e-4 relative error at the expert->worker partition boundary.
//
// Sized by WP_DISPATCH_UNION on 2026-08-05, config of record, 659-token prefill:
// the two workers behind the 1 GbE need 68.0% / 67.9% of the rows they are sent,
// so 263 MB of the 820.6 MB crossing the wire per prefill is removable. The
// loopback worker needs 99.8% and is left on the identity path, which is the
// right outcome -- bytes are cheap there and the gather would be pure overhead.
//
// Read once at startup; a per-request getenv on the dispatch path would itself
// distort the thing being measured.
const bool s_gather = [] {
    const char * value = std::getenv("WP_DISPATCH_GATHER");
    return value == nullptr || std::strcmp(value, "0") != 0;
}();

// WP_DISPATCH_GATHER_MAX_FRAC: gather only when a worker needs at most this
// fraction of the rows. Default 0.90 -- see the note at the use site for why a
// bare "needs fewer than all" test is not enough (it fires at 658 of 659 rows
// for the R9700 and costs more than it saves). 1.0 restores the old behaviour.
const double s_gather_max_frac = [] {
    const char * value = std::getenv("WP_DISPATCH_GATHER_MAX_FRAC");
    if (value == nullptr || value[0] == '\0') {
        return 0.90;
    }
    const double parsed = std::atof(value);
    return parsed > 0.0 && parsed <= 1.0 ? parsed : 0.90;
}();

// WP_SLICE_SKIP_SCAN=1: on a SLICED layer, skip the touched-token union scan in
// plan_requests. In slice mode every covering worker holds every expert, so the
// broadcast puts the FULL assignment set on every worker and top-k routing
// guarantees every token has a nonzero-weight expert -- the scan therefore
// yields needed.size()==n_tokens every time (compact branch never taken), so it
// is pure spine CPU on the prefill critical path. Default OFF (opt-in for A/B);
// forced off whenever WP_DISPATCH_UNION is measuring so its per-worker union log
// still runs. Numerically a no-op in slice mode.
const bool s_slice_skip_scan = [] {
    const char * value = std::getenv("WP_SLICE_SKIP_SCAN");
    return value != nullptr && std::strcmp(value, "1") == 0;
}();

// WP_SLICE_ENCODE_ONCE=1: on a SLICED layer, build + encode the request frame
// ONCE (for the first covering worker) and reuse the bytes for the rest. In
// slice mode the covering workers' frames are byte-identical (same layer,
// n_tokens, assignments, activations, swiglu_clamp, seq_id -- no per-worker
// field on the wire), so re-deriving wire_request and re-encoding per worker is
// redundant. Default OFF (opt-in for A/B); disabled under WP_DISPATCH_UNION so
// its per-worker diagnostic still runs. Numerically a no-op (identical bytes).
const bool s_slice_encode_once = [] {
    const char * value = std::getenv("WP_SLICE_ENCODE_ONCE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}();

// WP_DISPATCH_DEDUP_ACTIVATIONS=1: DEFAULT OFF. See the PIPE_VERSION 14 comment
// block in pipe-protocol.h and dedup_publish_and_ref() below for the full
// design. One-line version: on a SLICED layer wide enough to matter, two or
// more workers on the SAME machine (worker_info::machine) need the identical
// full activation tensor -- measured 2026-08-19, s1+s2 on 192.168.1.33 each
// pulled their own 1633 MB copy of one 2322-token prefill's activations over
// the SAME 1 GbE link, 3266 MB for bytes that only needed to cross the wire
// once. This sends the tensor to ONE worker on the machine and has it publish
// to a local POSIX shm segment the rest read directly, instead of paying the
// wire cost twice.
const bool s_dedup_activations = [] {
    const char * value = std::getenv("WP_DISPATCH_DEDUP_ACTIVATIONS");
    return value != nullptr && std::strcmp(value, "1") == 0;
}();

// WP_DISPATCH_DEDUP_MIN_TOKENS: only engage dedup when n_tokens exceeds this.
// Default 32, matching WP_DEFER_MAX_WIDTH's decode/prefill boundary elsewhere
// in this file. Decode (a spec-verify batch, a few tokens wide) gains nothing
// -- the remote workers already finish before the local one and their wire
// time is hidden -- and every synchronization this mechanism adds (the
// publish-ack round trip, the shm rendezvous) is pure downside on a path this
// latency-sensitive. Prefill (hundreds to thousands of tokens) is where the
// multi-megabyte-per-worker cost above actually lives.
const uint32_t s_dedup_min_tokens = [] {
    const char * value = std::getenv("WP_DISPATCH_DEDUP_MIN_TOKENS");
    if (value == nullptr || value[0] == '\0') {
        return (uint32_t) 32;
    }
    const long parsed = std::atol(value);
    return parsed > 0 ? (uint32_t) parsed : (uint32_t) 32;
}();

// WP_ISSUE_WIDEST_FIRST=1: order the per-layer wire SEND so the widest-slice
// worker (most expert-compute, the long pole) is issued earliest. Bit-exact:
// only the send/enqueue order changes; planning, harvest, and the fixed-order
// partial fold are untouched (see issue_order + issue_requests). Default OFF.
const bool s_issue_widest_first = [] {
    const char * value = std::getenv("WP_ISSUE_WIDEST_FIRST");
    return value != nullptr && std::strcmp(value, "1") == 0;
}();

// WP_AWAIT_SPIN_US = microseconds to spin on poll(fd, POLLIN, 0) before
// entering the blocking recv in await_response(). Default 0 (OFF).
//
// Every pipeline socket is blocking and nothing sets SO_BUSY_POLL, so the
// dispatch thread parks in recv() for the whole worker service time and has to
// be brought back by a normal socket wakeup: softirq -> runqueue -> (possibly)
// a C-state exit on an otherwise idle core. That wakeup is tens of
// microseconds at best and can be far worse on an idling box, it happens on
// EVERY awaited request, and it lands squarely in the unattributed gap between
// what the worker's own clock reports and what the spine measures as wait --
// the worker cannot see it and the spine cannot separate it from wire time.
// Spinning on a zero-timeout poll() for the tail of the wait means the bytes
// are consumed by a thread that is already on-core.
//
// Bit-exact: this only changes WHEN we call recv, never what is received, in
// what order requests are awaited, or in what order partials are folded. On
// timeout it falls through to exactly the same blocking pipe_recv_frame the
// unspun path uses, so behaviour on the slow path is unchanged too.
//
// Cost: it burns one core for up to the configured window per request. The
// useful setting is a little above the observed per-leg service time (worker
// service is ~0.8 ms/req today), and a value that large is only sane because
// the spine has nothing else to do while blocked. Start small (e.g. 200) to
// price the wakeup itself before paying for the whole service window.
const uint32_t s_await_spin_us = [] {
    const char * value = std::getenv("WP_AWAIT_SPIN_US");
    if (value == nullptr || value[0] == '\0') {
        return (uint32_t) 0;
    }
    const long parsed = std::atol(value);
    return parsed > 0 ? (uint32_t) parsed : (uint32_t) 0;
}();

// WP_DEFER_K = number of experts computed immediately per token.
// Unset / empty / non-positive => feature off (defer nothing).
int parse_wp_defer_k() {
    const char * value = std::getenv("WP_DEFER_K");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > 1000000L) {
        return 0;
    }
    return (int) parsed;
}

int dispatch_chunks_enabled() {
    const char * value = std::getenv("WP_DISPATCH_CHUNKS");
    if (value == nullptr || value[0] == '\0') {
        return 1;
    }
    const int parsed = std::atoi(value);
    return parsed == 2 ? 2 : 1;
}

// WP_DEFER_MAX_WIDTH = upper bound on n_tokens for a dispatch to be eligible
// for WP_DEFER_K deferral. Default 32.
//
// The gate used to be n_tokens == 1, meant to mean "this is decode, not
// prefill". With speculative decoding ON, a decode forward pass is a VERIFY
// BATCH of 1 + up to 7 draft tokens, never a single token -- so n_tokens == 1
// was never true here and the whole WP_DEFER_K mechanism was dead code in
// production. A batch wider than one token is not thereby a prompt: prefill
// dispatches the full ubatch (2048 tokens) in one shot, while the decode/
// spec-verify window is at most a couple dozen tokens, so width -- not
// exact-one -- is the real prefill/decode discriminator. This is the third
// time this same width/prompt conflation has been found in this codebase.
uint32_t defer_max_width_enabled() {
    const char * value = std::getenv("WP_DEFER_MAX_WIDTH");
    if (value == nullptr || value[0] == '\0') {
        return 32;
    }
    const int n = std::atoi(value);
    return n < 0 ? 0u : (uint32_t) n;
}

uint64_t elapsed_ns(dispatch_clock::time_point begin, dispatch_clock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

std::string endpoint_label(const endpoint & value) {
    return value.host + ":" + std::to_string(value.port);
}

std::string assignment_experts(const std::vector<pipe_expert_assignment> & assignments) {
    std::ostringstream stream;
    for (size_t i = 0; i < assignments.size(); ++i) {
        if (i != 0) {
            stream << ",";
        }
        stream << assignments[i].expert_id;
    }
    return stream.str();
}

// Sum of io_ticks (ms) over all whole-disk nvme devices (not partitions).
// Field layout matches /tmp/qd_sample.py and the kernel docs:
//   0-based tokens: [0]=major [1]=minor [2]=name ... [12]=io_ticks [13]=weighted_ms
// Returns false if none found or /proc/diskstats unreadable.
bool sample_nvme_io_ticks(uint64_t & io_ticks_ms_sum) {
    std::ifstream in("/proc/diskstats");
    if (!in) {
        return false;
    }
    io_ticks_ms_sum = 0;
    bool any = false;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::vector<std::string> tok;
        std::string              t;
        while (fields >> t) {
            tok.push_back(t);
        }
        if (tok.size() < 13) {
            continue;
        }
        const std::string & name = tok[2];
        if (name.rfind("nvme", 0) != 0) {
            continue;
        }
        // Whole devices: nvme0n1. Partitions: nvme0n1p1 / nvme0n1p2 — skip.
        // Match "p" only after the namespace digit so we do not reject nvme0n1.
        {
            const size_t npos = name.find('n');
            if (npos != std::string::npos && name.find('p', npos) != std::string::npos) {
                continue;
            }
        }
        // io_ticks is token[12] (1-based field 13). Do NOT read 11 values after
        // name and then one more — that lands on token[14], which is nearly
        // static under pure-read loads and yields util% ≈ 0.0.
        char * end = nullptr;
        const unsigned long long io_ticks = std::strtoull(tok[12].c_str(), &end, 10);
        if (end == tok[12].c_str() || *end != '\0') {
            continue;
        }
        io_ticks_ms_sum += (uint64_t) io_ticks;
        any = true;
    }
    return any;
}

// Split assignments into immediate (top K by router weight per token) and deferred.
// An expert may appear in both with complementary per-token weight masks.
// When defer_k <= 0, all experts are immediate.
void split_immediate_deferred(const std::vector<pipe_expert_assignment> & assignments,
                              uint32_t                                   n_tokens,
                              int                                        defer_k,
                              std::vector<pipe_expert_assignment> &      immediate,
                              std::vector<pipe_expert_assignment> &      deferred,
                              size_t &                                   n_deferred_count) {
    immediate.clear();
    deferred.clear();
    n_deferred_count = 0;

    if (defer_k <= 0 || assignments.empty()) {
        immediate = assignments;
        return;
    }

    // per_token_immediate[token] = set of expert ids that are immediate for that token
    std::vector<std::set<int32_t>> per_token_immediate((size_t) n_tokens);

    for (uint32_t token = 0; token < n_tokens; ++token) {
        struct ranked {
            int32_t expert_id;
            float   weight;
        };
        std::vector<ranked> ranked_experts;
        ranked_experts.reserve(assignments.size());
        for (const pipe_expert_assignment & assignment : assignments) {
            const float w = assignment.weights[(size_t) token];
            if (w != 0.0f) {
                ranked_experts.push_back({ assignment.expert_id, w });
            }
        }
        std::stable_sort(ranked_experts.begin(), ranked_experts.end(),
                         [](const ranked & a, const ranked & b) {
                             if (a.weight != b.weight) {
                                 return a.weight > b.weight;
                             }
                             return a.expert_id < b.expert_id;
                         });
        const size_t keep = std::min((size_t) defer_k, ranked_experts.size());
        for (size_t i = 0; i < keep; ++i) {
            per_token_immediate[(size_t) token].insert(ranked_experts[i].expert_id);
        }
    }

    for (const pipe_expert_assignment & assignment : assignments) {
        pipe_expert_assignment imm;
        pipe_expert_assignment def;
        imm.expert_id = assignment.expert_id;
        def.expert_id = assignment.expert_id;
        imm.weights.assign((size_t) n_tokens, 0.0f);
        def.weights.assign((size_t) n_tokens, 0.0f);
        bool any_imm = false;
        bool any_def = false;
        for (uint32_t token = 0; token < n_tokens; ++token) {
            const float w = assignment.weights[(size_t) token];
            if (w == 0.0f) {
                continue;
            }
            if (per_token_immediate[(size_t) token].count(assignment.expert_id) != 0) {
                imm.weights[(size_t) token] = w;
                any_imm                     = true;
            } else {
                def.weights[(size_t) token] = w;
                any_def                     = true;
            }
        }
        if (any_imm) {
            immediate.push_back(std::move(imm));
        }
        if (any_def) {
            deferred.push_back(std::move(def));
            ++n_deferred_count;
        }
    }

    // Safety: never leave a token with zero immediate experts when K >= 1.
    // (Can happen if K is set but a token selected fewer than K — already handled
    // by keep = min(K, size). If all went deferred somehow, fall back.)
    if (immediate.empty() && !assignments.empty()) {
        immediate = assignments;
        deferred.clear();
        n_deferred_count = 0;
    }
}

// A fully-encoded wire frame awaiting a writer thread. Only bytes are moved;
// the request itself (assignments / token_ids) lives on the dispatch thread and
// is untouched by the writer. seq_id is echoed so a writer failure can be
// reported against the request sequence without decoding the payload.
struct wire_frame {
    pipe_frame_type      type = PIPE_HELLO;
    uint64_t             seq_id = 0;
    int32_t              layer = -1;
    std::vector<uint8_t> payload;
    dispatch_clock::time_point enqueued_at{};
};

}  // namespace

struct dispatcher::impl {
    static constexpr size_t speed_estimate_min_samples = 3;
    static constexpr size_t speed_estimate_window       = 8;
    static constexpr size_t speed_estimate_min_spread   = 2;

    struct speed_sample {
        size_t n = 0;
        double wait_ms = 0.0;
    };

    // D1 async issue: one FIFO writer thread per socket. The dispatch thread
    // enqueues fully-encoded frames and returns; the writer moves bytes. A
    // single writer per socket is what keeps frame bytes from interleaving on
    // the wire -- do NOT pool sockets across writer threads.
    struct socket_writer {
        pipe_socket_ptr    socket;   // own ref: poison()'s reset cannot free it mid-send
        std::string        endpoint;
        std::thread        thread;
        std::mutex         mutex;
        std::condition_variable cv;
        std::deque<wire_frame> queue;
        bool               stop    = false;
        bool               done    = false;
        bool               failed  = false;
        std::string        error_msg;
        // Backpressure cap. One layer of requests is <= ~3 frames per worker
        // (<= ~60 MB worst-case prefill), so a full queue means a wedged worker;
        // fail loudly rather than grow without bound.
        static constexpr size_t MAX_QUEUE = 8;
    };

    // WP_CONCURRENT_ISSUE: one persistent sender thread per socket, but unlike
    // socket_writer above there is no queue -- a single job slot, posted then
    // joined by issue_requests() before it returns. See concurrent_issue_enabled()
    // for why this is a different shape from WP_ASYNC_ISSUE, not a copy of it.
    struct concurrent_sender {
        pipe_socket_ptr    socket;   // own ref, same reasoning as socket_writer's
        std::thread        thread;
        std::mutex         mutex;
        std::condition_variable cv;
        bool               stop      = false;
        bool               has_job   = false;
        bool               job_done  = true;
        bool               ok        = true;
        // Job description. Valid only while has_job is true; points directly
        // into the posting request's payload vectors (no copy) which outlive
        // the join because issue_requests() does not return until job_done.
        pipe_frame_type    type1 = PIPE_HELLO;
        pipe_frame_type    type2 = PIPE_HELLO;
        bool               has_second = false;
        uint64_t           seq_id = 0;
        const uint8_t *     data1 = nullptr;
        size_t              len1  = 0;
        const uint8_t *     data2 = nullptr;
        size_t              len2  = 0;
    };

    struct worker {
        endpoint                                 target;
        worker_info                              info;
        pipe_expert_hello                        hello;
        std::unique_ptr<inproc_backend>          inproc;
        pipe_socket_ptr                          socket;
        std::unique_ptr<socket_writer>           writer;
        std::unique_ptr<concurrent_sender>       sender;
        // D9: residency LRU as std::list + unordered_map (key = layer<<32|expert)
        // instead of an O(n_slots) vector memmove per assignment. Dispatch-thread
        // only -- the writer threads never touch it.
        std::list<std::pair<int32_t, int32_t>>   resident_lru;
        std::unordered_map<uint64_t, std::list<std::pair<int32_t, int32_t>>::iterator> resident_map;
        std::array<speed_sample, speed_estimate_window> speed_history{};
        size_t                                   speed_history_next      = 0;
        size_t                                   speed_samples            = 0;
        size_t                                   speed_n_spread            = 0;
        double                                   estimated_fixed_ms        = 0.0;
        double                                   estimated_ms_per_expert   = 0.0;
        bool                                     speed_fit_valid            = false;
    };

    struct planned_request {
        int32_t                             layer = -1;
        size_t                              worker_index = 0;
        std::vector<pipe_expert_assignment> assignments;
        std::vector<uint8_t>                payload;
        std::vector<uint8_t>                begin_payload;
        std::vector<uint8_t>                acts_payload;
        // PER-REQUEST split-frame decision (2026-08-27). WP_SPLIT_FRAME used to
        // be a connection-lifetime latch, so EVERY request paid two frames --
        // but split-frame exists only to carry WP_DISPATCH_DEDUP_ACTIVATIONS,
        // and dedup is itself gated to n_tokens > WP_DISPATCH_DEDUP_MIN_TOKENS
        // (see dedup_publish_and_ref). Decode therefore paid the second frame
        // to enable a mechanism that then declined to run.
        //
        // THROUGHPUT CLAIM RETRACTED 2026-08-28. An earlier version of this
        // comment recorded "decode 6.88 -> 7.51 tok/s (+9.1%)" for latching
        // split-frame off. That number is WITHDRAWN: it was measured across
        // two spine loads, and the spine has a decode warm-up curve (~175
        // ms/tok cold falling to ~128 ms/tok as cumulative decode crosses
        // ~1200-1700 tokens) that confounded every A/B run that night. The
        // paired prefill figure (79.5 -> 71.1 tok/s) is confounded the same
        // way. Do not cite either number; do not re-derive a lever from them.
        //
        // This gate is kept on STRUCTURAL grounds only, which stand on their
        // own: below the dedup threshold the second frame carries a mechanism
        // that then declines to run, so it is a wasted packet per request on a
        // TCP_NODELAY socket. Its throughput effect is UNMEASURED. Any future
        // measurement must warm both arms past ~1700 decode tokens since the
        // last worker cycle and state the warm-up position alongside the number.
        //
        // Set from the same n_tokens threshold dedup uses, so prefill frames
        // are byte-identical to the latched behaviour and only decode changes.
        bool                                split_wire = false;
        dispatch_clock::time_point          issued_at;
        dispatch_clock::time_point          await_started_at;
        dispatch_clock::time_point          await_finished_at;
        uint64_t                            response_bytes = 0;
        uint64_t                            unpack_ns = 0;
        uint64_t                            wait_ns = 0;
        // SPINE-SIDE GATHER (2026-08-05). When non-empty, this request carries
        // only these token rows -- token_ids[r] is the ORIGINAL token index of
        // compacted row r -- and the returned partial is [token_ids.size() x
        // n_embd], which must be SCATTERED back through this map.
        //
        // Empty means identity: the request carries all n_tokens rows and the
        // partial sums in directly. Decode (n_tokens == 1) and any worker that
        // genuinely needs every row take this path, so they are bit-for-bit
        // unchanged.
        //
        // Deliberately NOT on the wire. The worker is oblivious: its assignments'
        // weight vectors are compacted by the same map, so it just sees a
        // narrower batch and returns a narrower partial. That keeps PIPE_VERSION
        // at 4 and means no worker rebuild.
        std::vector<uint32_t>               token_ids;

        // WP_DISPATCH_DEDUP_ACTIVATIONS (see dedup_publish_and_ref()). Set only
        // for requests this mechanism sent as PIPE_EXPERT_DISPATCH_ACTS_REF
        // instead of a normal ACTS. already_issued tells issue_requests() this
        // request's BEGIN+ACTS(-variant) pair was already put on the wire by
        // dedup_publish_and_ref() -- issue_requests must not send it again.
        bool                                 already_issued        = false;
        bool                                 dedup_role_secondary  = false;
        // At most one retry per request; see receive_partial's fallback branch.
        bool                                 dedup_retried         = false;
        // Ordinary (non-dedup) ACTS payload for the SAME activations this
        // request's ACTS_REF pointed at, built once at plan time so the
        // fallback in receive_partial never has to re-touch `activations` or
        // re-derive anything -- it resends literally the bytes this request
        // would have carried had dedup never been attempted.
        std::vector<uint8_t>                dedup_fallback_acts_payload;

        // In-process trunk workers: skip encode/TCP. Dispatch runs in
        // await/finish, not issue — HIP graph capture is still open then.
        pipe_expert_dispatch_req            inproc_wire;
    };

    using dispatch_handle = dispatcher::dispatch_handle;
    static constexpr dispatch_handle k_invalid_handle = 0;

    struct dispatch_state {
        dispatch_handle              handle = k_invalid_handle;
        int32_t                      layer = -1;
        uint32_t                     chunk_index = 0;
        uint32_t                     n_tokens = 0;
        uint64_t                     seq_id = 0;
        uint64_t                     activation_count = 0;
        uint64_t                     dispatch_hash_seq_ = 0;
        uint64_t                     hash_pre = 0;
        size_t                       hash_reqs = 0;
        dispatch_stats                stats;
        std::vector<planned_request>  imm_requests;
        std::vector<planned_request>  def_requests;
        std::vector<float>            folded_prev;
        std::vector<size_t>           assigned_counts;
        dispatch_clock::time_point    wait_start{};
        dispatch_clock::time_point    req_dispatch_start_{};
    };

    struct temporal_layer_stats {
        std::vector<int32_t>           previous_experts;
        uint64_t                       n_pairs = 0;
        uint64_t                       sum_overlap = 0;
        std::array<uint64_t, 9>        hist{};
    };

    mutable std::mutex                   layer_trace_mutex_;
    std::map<int32_t, layer_trace_stats> layer_traces_;

    void reset_layer_trace(int32_t layer) {
        if (!layer_trace_enabled()) {
            return;
        }
        std::lock_guard<std::mutex> lock(layer_trace_mutex_);
        layer_traces_[layer] = {};
    }

    void add_layer_trace(int32_t layer, uint64_t layer_trace_stats::* field, uint64_t ns) {
        if (!layer_trace_enabled()) {
            return;
        }
        std::lock_guard<std::mutex> lock(layer_trace_mutex_);
        layer_traces_[layer].*field += ns;
    }

    layer_trace_stats layer_trace(int32_t layer) const {
        if (!layer_trace_enabled()) {
            return {};
        }
        std::lock_guard<std::mutex> lock(layer_trace_mutex_);
        const auto it = layer_traces_.find(layer);
        return it == layer_traces_.end() ? layer_trace_stats{} : it->second;
    }

    // Deferred requests issued at layer N, collected at layer N+1's dispatch.
    struct pending_deferred_batch {
        int32_t                      layer    = -1;
        uint64_t                     seq_id   = 0;
        uint32_t                     n_tokens = 0;
        std::vector<planned_request> requests;
        // Set when the successor layer begins collecting — anything still
        // outstanding after the successor has already returned is late.
        bool                         fold_opened = false;
        bool                         fold_closed = false;
    };

    std::vector<worker>                                 workers;
    std::vector<worker_info>                            public_workers;
    // WP_ISSUE_WIDEST_FIRST: indices into `workers`, widest expert-range first.
    // Used ONLY to order the wire send in issue_requests; planning/harvest/fold
    // all keep worker-registration order, so the summed result is bit-exact.
    std::vector<size_t>                                 issue_order;
    std::map<int32_t, std::vector<std::vector<size_t>>> routes;
    std::map<std::string, size_t>                       machine_cursor;
    std::map<int32_t, temporal_layer_stats>              temporal_layers;
    dispatch_stats                                      last_stats;
    deferral_stats                                      deferral;
    prefetch_hint_stats                                 hint_stats;
    pending_deferred_batch                              pending_def;
    size_t                                              in_flight     = 0;
    int32_t                                             n_embd        = 0;
    int32_t                                             n_ff_exp      = 0;
    int32_t                                             n_expert      = 0;
    int32_t                                             n_expert_used = 0;
    int32_t                                             last_routed_layer = -1;
    // Host-provided last main-graph MoE layer that must not defer (no successor).
    // -1 => fall back to last_routed_layer from worker HELLO. Must be set to
    // hparams.n_layer()-1 so NextN/MTP layers (e.g. blk.78) advertised by
    // workers are not mistaken for the fold successor of the main stack.
    int32_t                                             last_no_defer_layer = -1;
    // Slice workers each retain every expert. Their per-expert partials are
    // linearly summed by the existing dispatcher accumulator.
    //
    // PER LAYER, not global. A mixed fleet (e.g. DSpark layers 43..45
    // withdrawn from the four GPU slice workers and served instead by two
    // full-width classic CPU workers, while layers 0..42 stay sliced across
    // the GPU fleet) means the mode is a property of the LAYER's own
    // covering-worker set, not of the dispatcher's worker list as a whole.
    // Populated by build_routes() alongside `routes`, so a layer present in
    // one map is present in the other; consumers look it up with .at(layer)
    // rather than re-deriving it.
    std::map<int32_t, bool>                             layer_slice_mode;
    int                                                 defer_k_value = 0;
    uint32_t                                            defer_max_width_ = 32;
    std::string                                         model_identity;
    bool                                                poisoned = false;
    bool                                                collect_stats = false;
    bool                                                speed_split   = false;
    // WP_DISPATCH_STATIC_ASSIGN, latched once per dispatcher. Read by BOTH
    // choose_worker and send_prefetch_hints -- one field so a hint can never be
    // routed by a different rule than the request that follows it.
    bool                                                static_assign = true;
    bool                                                hint_inflight = false;
    bool                                                stats_logging = false;
    // WP_ASYNC_ISSUE latch (D1). True => issue/hint frames go through per-socket
    // writer threads instead of blocking send() on the dispatch thread.
    bool                                                async_issue = false;
    // WP_CONCURRENT_ISSUE latch. See concurrent_issue_enabled() for the design
    // note. Forced off when async_issue is also requested (mutually exclusive:
    // both would try to put THIS request's frames on the wire through a
    // different mechanism) -- enforced in the constructor, not on the dispatch
    // path.
    bool                                                concurrent_issue = false;
    // WP_UNPACK_OVERLAP latch. See unpack_overlap_enabled() for the full
    // design note. Latched once, like the other flags above, so a per-layer
    // getenv never appears on the dispatch path.
    bool                                                unpack_overlap = false;
    bool                                                split_frame = false;
    // WP_DISPATCH_DEDUP_ACTIVATIONS latch. Requires split_frame (the mechanism
    // extends the BEGIN/ACTS split with two more per-machine-role ACTS
    // variants) and is incompatible with async_issue's writer-thread queue --
    // dedup_publish_and_ref() does its own synchronous send+recv of the
    // publish ack on the dispatch thread, which a writer thread could reorder
    // against. Both are enforced where dedup_activations is latched, in the
    // constructor, so a request never has to re-check them.
    bool                                                dedup_activations = false;
    uint32_t                                            dedup_min_tokens_ = 32;
    int                                                 dispatch_chunks_ = 1;
    // machine -> indices into `workers` sharing that machine, precomputed once
    // (machine membership is a property of the worker set, not of any one
    // dispatch). Only machines with >= 2 workers matter to dedup; kept as a
    // plain map indexed by machine string, filtered per-layer against which of
    // those workers are actually covering the layer being dispatched.
    std::map<std::string, std::vector<size_t>>          machine_workers_;
    // Per-worker static-assign weight (D6.3), default 1 = current behaviour.
    // WP_DISPATCH_WEIGHTS="port=w[,port=w...]" and the sugar
    // WP_DISPATCH_BIAS_1070=N (port 8803). Indexed by workers[] index.
    std::vector<int>                                    worker_weights_;
    int                                                 decode_prefer_port_ = 8803;
    uint32_t                                            decode_max_tokens_  = 8;
    uint64_t                                            temporal_n_pairs = 0;
    uint64_t                                            temporal_sum_overlap = 0;
    std::array<uint64_t, 9>                             temporal_hist{};
    FILE *                                              routing_dump_ =
        s_routing_dump_path != nullptr ? fopen(s_routing_dump_path, "a") : nullptr;
    int32_t                                             routing_dump_last_layer_ = -1;
    uint64_t                                            routing_dump_step_ = 0;

    // Per-request wire log; see accumulate_partial. Off unless WP_DISPATCH_REQ_LOG
    // is set, so it costs nothing in a normal run.
    FILE *                                              req_log_ = [] {
        const char * p = std::getenv("WP_DISPATCH_REQ_LOG");
        return (p != nullptr && p[0] != '\0') ? fopen(p, "w") : (FILE *) nullptr;
    }();
    FILE *                                              writer_log_ = [] {
        const char * p = std::getenv("WP_DISPATCH_REQ_LOG");
        return (p != nullptr && p[0] != '\0') ? fopen((std::string(p) + ".writer").c_str(), "w") : (FILE *) nullptr;
    }();
    std::mutex                                          writer_log_mutex_;

    // Gap accounting: time spent with in_flight == 0.
    bool                       gap_at_zero = false;
    dispatch_clock::time_point gap_zero_since{};
    bool                       window_active = false;
    uint64_t                   window_io_ticks_begin = 0;
    dispatch_clock::time_point window_wall_begin{};
    bool                       window_sample_ok = false;

    explicit impl(const std::vector<endpoint> & endpoints) :
                speed_split(speed_split_enabled()),
                static_assign(static_assign_enabled()),
                hint_inflight(hint_inflight_enabled()),
                async_issue(async_issue_enabled()),
                unpack_overlap(unpack_overlap_enabled()),
                dispatch_chunks_(dispatch_chunks_enabled()) {
        if (dispatch_chunks_ > 1) {
            if (!static_assign) {
                static_assign = true;
                LLAMA_LOG_WARN(
                             "expert dispatch: WP_DISPATCH_STATIC_ASSIGN=0 disabled while dispatch chunks are active\n");
            }
            if (async_issue) {
                async_issue = false;
                LLAMA_LOG_WARN(
                             "expert dispatch: WP_ASYNC_ISSUE=1 disabled while dispatch chunks are active\n");
            }
            if (s_dedup_activations) {
                LLAMA_LOG_WARN(
                             "expert dispatch: WP_DISPATCH_DEDUP_ACTIVATIONS=1 disabled while dispatch chunks are active\n");
            }
            if (const char * overlap = std::getenv("WP_EXPERT_OVERLAP");
                overlap != nullptr && overlap[0] == '1') {
                LLAMA_LOG_WARN(
                             "expert dispatch: WP_EXPERT_OVERLAP=1 is incompatible with chunked dispatch; "
                             "worker launch must set WP_EXPERT_OVERLAP=0\n");
            }
            LLAMA_LOG_WARN(
                         "expert dispatch: chunked dispatch requires WP_WORKER_PIPELINE=1 "
                         "in the worker launch environment; the spine cannot verify that setting\n");
        }
        decode_prefer_port_ = decode_prefer_port_enabled();
        if (unpack_overlap) {
            std::fprintf(stderr,
                         "expert dispatch: WP_UNPACK_OVERLAP=1 -- decoding worker partials as they "
                         "arrive (poll order), folding in fixed worker order (bit-exact); forward-"
                         "budget ns_wait may now include decode cost that used to show as ns_unpack\n");
        }
        concurrent_issue = concurrent_issue_enabled() && !async_issue;
        if (concurrent_issue_enabled() && !concurrent_issue) {
            std::fprintf(stderr,
                         "expert dispatch: WP_CONCURRENT_ISSUE requested but disabled "
                         "(mutually exclusive with WP_ASYNC_ISSUE)\n");
        }
        decode_max_tokens_  = decode_max_tokens_enabled();
        const int connect_retry_s = dispatch_connect_retry_seconds();
        const auto connect_deadline = dispatch_clock::now() + std::chrono::seconds(connect_retry_s);
        split_frame = split_frame_enabled();
        // See the field comment: dedup requires split_frame and is mutually
        // exclusive with async_issue by construction, not by a runtime check
        // on the dispatch path.
        dedup_activations = dispatch_chunks_ > 1 ? false : s_dedup_activations && split_frame && !async_issue;
        dedup_min_tokens_ = s_dedup_min_tokens;
        if (s_dedup_activations && !dedup_activations) {
            LLAMA_LOG_WARN(
                         "expert dispatch: WP_DISPATCH_DEDUP_ACTIVATIONS requested but disabled "
                         "(requires WP_SPLIT_FRAME=1 and WP_ASYNC_ISSUE unset)\n");
        }
        stats_logging = dispatch_stats_enabled();
        collect_stats = stats_logging || speed_split;
        defer_k_value     = parse_wp_defer_k();
        if (dispatch_chunks_ > 1 && defer_k_value > 0) {
            defer_k_value = 0;
            std::fprintf(stderr,
                         "expert dispatch: WP_DEFER_K disabled while dispatch chunks are active\n");
        }
        deferral.defer_k  = defer_k_value;
        defer_max_width_  = defer_max_width_enabled();

        if (endpoints.empty()) {
            throw std::invalid_argument("expert dispatcher requires at least one worker endpoint");
        }
        if (!pipe_transport_init()) {
            throw std::runtime_error("expert dispatcher failed to initialize TCP transport");
        }

        std::set<std::string> seen_endpoints;
        for (endpoint target : endpoints) {
            if (target.host.empty() || target.port <= 0 || target.port > 65535) {
                throw std::invalid_argument("expert dispatcher has an invalid worker endpoint");
            }
            if (target.machine.empty()) {
                target.machine = target.host;
            }
            const std::string label = endpoint_label(target);
            if (!seen_endpoints.insert(label).second) {
                throw std::invalid_argument("expert dispatcher repeats worker " + label);
            }

            worker connected;
            connected.target        = target;
            connected.info.endpoint = label;
            connected.info.machine  = target.machine;
            if (g_inproc_factory != nullptr) {
                connected.inproc = g_inproc_factory(target);
            }
            if (connected.inproc) {
                connected.hello = connected.inproc->hello();
                std::fprintf(stderr,
                             "expert dispatch: in-process worker %s experts=%d..%d slots=%u layers=%zu\n",
                             label.c_str(),
                             connected.hello.expert_first, connected.hello.expert_last,
                             connected.hello.n_slots, connected.hello.layers.size());
            } else {
            bool retryable_connect = false;
            do {
                connected.socket = pipe_socket_t::connect(target.host.c_str(), target.port, &retryable_connect);
                if (connected.socket || !retryable_connect || connect_retry_s == 0 ||
                    dispatch_clock::now() >= connect_deadline) {
                    break;
                }
                const auto remaining = std::chrono::duration_cast<std::chrono::seconds>(
                    connect_deadline - dispatch_clock::now()).count();
                std::fprintf(stderr, "expert dispatch: waiting for workers (%llds left)\n",
                             (long long) std::max<int64_t>(0, remaining));
                const auto retry_delay = std::chrono::duration_cast<dispatch_clock::duration>(
                    std::chrono::seconds(2));
                std::this_thread::sleep_for(std::min(retry_delay,
                                                      connect_deadline - dispatch_clock::now()));
            } while (true);
            if (!connected.socket) {
                throw std::runtime_error("expert dispatcher failed to connect to worker " + label);
            }

            pipe_frame_type      type;
            uint64_t             seq_id = 0;
            std::vector<uint8_t> payload;
            if (!pipe_recv_frame(*connected.socket, type, seq_id, payload)) {
                throw std::runtime_error("expert dispatcher worker " + label + " died before HELLO");
            }
            if (type == PIPE_ERROR) {
                const pipe_error error = pipe_decode_error(payload.data(), payload.size());
                throw std::runtime_error("expert dispatcher worker " + label + " rejected HELLO: " + error.msg);
            }
            if (type != PIPE_HELLO || seq_id != 0) {
                throw std::runtime_error("expert dispatcher worker " + label + " sent an invalid HELLO frame");
            }
            try {
                connected.hello = pipe_decode_expert_hello(payload.data(), payload.size());
            } catch (const std::exception & error) {
                throw std::runtime_error("expert dispatcher worker " + label +
                                         " has an invalid HELLO: " + error.what());
            }
            if (connected.hello.role != PIPE_EXPERT_ROLE_WORKER) {
                throw std::runtime_error("expert dispatcher peer " + label + " is not an expert worker");
            }
            }
            if (connected.hello.role != PIPE_EXPERT_ROLE_WORKER) {
                throw std::runtime_error("expert dispatcher peer " + label + " is not an expert worker");
            }

            if (workers.empty()) {
                n_embd         = connected.hello.n_embd;
                n_ff_exp       = connected.hello.n_ff_exp;
                n_expert       = connected.hello.n_expert;
                n_expert_used  = connected.hello.n_expert_used;
                model_identity = connected.hello.model_identity;
            } else if (connected.hello.n_embd != n_embd || connected.hello.n_ff_exp != n_ff_exp ||
                       connected.hello.n_expert != n_expert || connected.hello.n_expert_used != n_expert_used ||
                       connected.hello.model_identity != model_identity) {
                throw std::runtime_error("expert dispatcher worker " + label +
                                         " does not match the first worker's model identity and hparams");
            }

            connected.info.expert_first   = connected.hello.expert_first;
            connected.info.expert_last    = connected.hello.expert_last;
            connected.info.n_slots        = connected.hello.n_slots;
            connected.info.layers         = connected.hello.layers;
            connected.info.shard_identity = connected.hello.shard_identity;

            if (!connected.inproc) {
            pipe_expert_hello client = connected.hello;
            client.role              = PIPE_EXPERT_ROLE_CLIENT;
            client.expert_first      = -1;
            client.expert_last       = -1;
            client.n_slots           = 0;
            client.layers.clear();
            std::vector<uint8_t> payload = pipe_encode_expert_hello(client);
            if (!pipe_send_frame(*connected.socket, PIPE_HELLO, 0, payload.data(), payload.size())) {
                throw std::runtime_error("expert dispatcher failed to send HELLO to worker " + label);
            }

            pipe_frame_type type;
            uint64_t        seq_id = 0;
            if (!pipe_recv_frame(*connected.socket, type, seq_id, payload)) {
                throw std::runtime_error("expert dispatcher worker " + label + " died during HELLO");
            }
            if (type != PIPE_EXPERT_HELLO_ACK || seq_id != 0) {
                throw std::runtime_error("expert dispatcher worker " + label +
                                         " sent an invalid expert HELLO acknowledgement");
            }
            pipe_expert_hello_ack ack;
            try {
                ack = pipe_decode_expert_hello_ack(payload.data(), payload.size());
            } catch (const std::exception & error) {
                throw std::runtime_error("expert dispatcher worker " + label +
                                         " sent an invalid expert HELLO acknowledgement: " + error.what());
            }
            if (!ack.accepted) {
                throw std::runtime_error("expert dispatcher worker " + label +
                                         " rejected HELLO: " + ack.reason);
            }
            }

            public_workers.push_back(connected.info);
            workers.push_back(std::move(connected));
        }

        // WP_ISSUE_WIDEST_FIRST: precompute a widest-slice-first SEND order from
        // the width already advertised in HELLO (expert_last-expert_first+1). A
        // malformed/never-set range sorts LAST (width 0). Ties break by original
        // registration index (stable) so the order is deterministic across runs.
        // Default (flag off) keeps registration order == the endpoint-string
        // order. Send order only; the partial fold stays registration-ordered.
        issue_order.resize(workers.size());
        std::iota(issue_order.begin(), issue_order.end(), (size_t) 0);
        if (s_issue_widest_first) {
            const auto width = [this](size_t i) -> int64_t {
                const int32_t f = workers[i].info.expert_first;
                const int32_t l = workers[i].info.expert_last;
                return (f >= 0 && l >= f) ? (int64_t) l - (int64_t) f + 1 : 0;
            };
            std::stable_sort(issue_order.begin(), issue_order.end(),
                             [&](size_t a, size_t b) { return width(a) > width(b); });
        }

        for (size_t i = 0; i < workers.size(); ++i) {
            machine_workers_[workers[i].info.machine].push_back(i);
        }
        build_routes();
        parse_worker_weights();
        if (decode_prefer_port_ > 0 && decode_max_tokens_ > 0) {
            std::fprintf(stderr,
                         "expert dispatch: decode n_tokens<=%u prefers port %d (WP_DISPATCH_DECODE_PORT)\n",
                         (unsigned) decode_max_tokens_, decode_prefer_port_);
        }
        start_writers();
        start_concurrent_senders();
    }

    // Normal destruction lets queued frames drain before joining writers.
    ~impl() {
        stop_writers(false);
        stop_concurrent_senders();
        if (req_log_ != nullptr) fclose(req_log_);
        if (writer_log_ != nullptr) fclose(writer_log_);
        if (routing_dump_ != nullptr) fclose(routing_dump_);
    }

    // Residency key (layer, expert) -> uint64 for the D9 LRU map.
    static uint64_t residency_key(int32_t layer, int32_t expert) {
        return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert;
    }

    // Parse WP_DISPATCH_WEIGHTS and WP_DISPATCH_BIAS_1070 into per-worker
    // static-assign weights (D6.3). Keyed by the port of each worker's endpoint
    // (host:port); default weight 1 = current uniform behaviour. A weight is
    // clamped to >= 1. Never throws: a malformed knob degrades to the default.
    void parse_worker_weights() {
        worker_weights_.assign(workers.size(), 1);
        auto apply_by_port = [this](int port, int weight) {
            if (port <= 0 || weight < 1) {
                return;
            }
            for (size_t i = 0; i < workers.size(); ++i) {
                const size_t colon = workers[i].info.endpoint.rfind(':');
                if (colon == std::string::npos) {
                    continue;
                }
                const int wport = std::atoi(workers[i].info.endpoint.c_str() + colon + 1);
                if (wport == port) {
                    worker_weights_[i] = std::max(worker_weights_[i], weight);
                }
            }
        };
        if (const char * w = std::getenv("WP_DISPATCH_WEIGHTS"); w != nullptr && w[0] != '\0') {
            std::istringstream stream(w);
            std::string        item;
            while (std::getline(stream, item, ',')) {
                const size_t eq = item.find('=');
                if (eq == std::string::npos || eq == 0) {
                    continue;
                }
                apply_by_port(std::atoi(item.substr(0, eq).c_str()),
                              std::atoi(item.substr(eq + 1).c_str()));
            }
        }
        if (const char * b = std::getenv("WP_DISPATCH_BIAS_1070"); b != nullptr && b[0] != '\0') {
            // Fleet convention: the GTX 1070 listens on 8803. Bias its weight so
            // severe-tail (RX 480) layers shift to the card that drains at
            // 2.9 GB/s. Pure function of (layer, expert) still -- see choose_worker.
            apply_by_port(8803, std::atoi(b));
        }
    }

    void start_writers() {
        if (!async_issue) {
            return;
        }
        try {
            for (worker & value : workers) {
                if (!value.socket) {
                    continue;
                }
                value.writer.reset(new socket_writer{});
                value.writer->socket = value.socket;   // own ref for the thread
                value.writer->endpoint = value.info.endpoint;
                socket_writer * w = value.writer.get();
                w->thread = std::thread([this, w]() { writer_loop(w); });
            }
        } catch (...) {
            // std::thread construction can throw (rare resource exhaustion). Stop
            // whatever writers did start so no thread is left running into impl
            // storage when the constructor unwinds. Threads already joined are a
            // no-op (not joinable).
            stop_writers(true);
            throw;
        }
    }

    void stop_writers(bool immediate) {
        for (worker & value : workers) {
            socket_writer * w = value.writer.get();
            if (!w) {
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(w->mutex);
                w->stop = true;
            }
            w->cv.notify_all();
        }
        const auto deadline = dispatch_clock::now() + std::chrono::seconds(3);
        if (!immediate) {
            for (worker & value : workers) {
                socket_writer * w = value.writer.get();
                if (!w) {
                    continue;
                }
                std::unique_lock<std::mutex> lock(w->mutex);
                w->cv.wait_until(lock, deadline, [w]() { return w->done; });
            }
        }
        for (worker & value : workers) {
            socket_writer * w = value.writer.get();
            if (!w) {
                continue;
            }
            std::lock_guard<std::mutex> lock(w->mutex);
            if (!w->done && w->socket) {
                w->socket->shutdown();
            }
        }
        for (worker & value : workers) {
            socket_writer * w = value.writer.get();
            if (!w) {
                continue;
            }
            if (w->thread.joinable()) {
                w->thread.join();
            }
            w->socket.reset();
        }
    }

    // Writer thread body: pop one encoded frame at a time and send it. On send
    // failure, record it and exit -- the dispatch thread observes the failure at
    // the next enqueue or await_response entry and poisons. Never throws.
    void writer_loop(socket_writer * w) {
        while (true) {
            wire_frame frame;
            bool       have = false;
            {
                std::unique_lock<std::mutex> lock(w->mutex);
                w->cv.wait(lock, [w]() {
                    return w->stop || w->failed || !w->queue.empty();
                });
                if (w->failed) {
                    break;
                }
                if (!w->queue.empty()) {
                    frame = std::move(w->queue.front());
                    w->queue.pop_front();
                    have = true;
                } else if (w->stop) {
                    break;
                }
            }
            if (!have) {
                continue;
            }
            const dispatch_clock::time_point send_started = dispatch_clock::now();
            const bool send_ok = pipe_send_frame(*w->socket, frame.type, frame.seq_id,
                                                 frame.payload.data(), frame.payload.size());
            if (layer_trace_enabled() && frame.layer >= 0) {
                add_layer_trace(frame.layer, &layer_trace_stats::send_ns,
                                elapsed_ns(send_started, dispatch_clock::now()));
            }
            if (writer_log_ != nullptr) {
                std::lock_guard<std::mutex> log_lock(writer_log_mutex_);
                fprintf(writer_log_, "%llu %llu %u %llu %zu %s\n",
                        (unsigned long long) elapsed_ns(frame.enqueued_at, send_started),
                        (unsigned long long) elapsed_ns(send_started, dispatch_clock::now()),
                        (unsigned) frame.type, (unsigned long long) frame.seq_id,
                        frame.payload.size(), w->endpoint.c_str());
                fflush(writer_log_);
            }
            if (!send_ok) {
                std::lock_guard<std::mutex> lock(w->mutex);
                if (!w->failed) {
                    w->failed     = true;
                    w->error_msg  = "writer send failed on seq_id " + std::to_string(frame.seq_id);
                }
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(w->mutex);
            w->done = true;
        }
        w->cv.notify_all();
    }

    // Enqueue one frame onto a worker's writer FIFO. Throws if the writer has
    // already failed or the queue is over its cap (a wedged worker).
    void enqueue_frame(worker & value, wire_frame frame) {
        socket_writer * w = value.writer.get();
        if (!w) {
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                     " has no writer (async issue off)");
        }
        {
            std::lock_guard<std::mutex> lock(w->mutex);
            if (w->failed) {
                throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                         " writer failed: " + w->error_msg);
            }
            if (w->queue.size() >= socket_writer::MAX_QUEUE) {
                throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                         " writer queue overflow (worker wedged?)");
            }
            frame.enqueued_at = dispatch_clock::now();
            w->queue.push_back(std::move(frame));
        }
        w->cv.notify_one();
    }

    void start_concurrent_senders() {
        if (!concurrent_issue) {
            return;
        }
        try {
            for (worker & value : workers) {
                if (!value.socket) {
                    continue;
                }
                value.sender.reset(new concurrent_sender{});
                value.sender->socket = value.socket;   // own ref, see field comment
                concurrent_sender * s = value.sender.get();
                s->thread = std::thread([this, s]() { concurrent_sender_loop(s); });
            }
        } catch (...) {
            stop_concurrent_senders();
            throw;
        }
    }

    void stop_concurrent_senders() {
        for (worker & value : workers) {
            concurrent_sender * s = value.sender.get();
            if (!s) {
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(s->mutex);
                s->stop = true;
            }
            s->cv.notify_all();
        }
        for (worker & value : workers) {
            concurrent_sender * s = value.sender.get();
            if (!s) {
                continue;
            }
            if (s->thread.joinable()) {
                s->thread.join();
            }
            s->socket.reset();
        }
    }

    // Sender thread body: block for a job, run its one or two blocking sends on
    // THIS thread (so this socket's frame order is exactly as posted, matching
    // the serial path's per-socket order), report completion, go back to
    // sleep. No queue: the poster (issue_requests, via join_concurrent_job)
    // never posts a socket's next job until it has joined this one, so there is
    // at most one job in flight per socket and nothing here can back up.
    void concurrent_sender_loop(concurrent_sender * s) {
        while (true) {
            std::unique_lock<std::mutex> lock(s->mutex);
            s->cv.wait(lock, [s]() { return s->stop || s->has_job; });
            if (!s->has_job) {
                // stop with no pending job
                break;
            }
            const pipe_frame_type type1 = s->type1;
            const pipe_frame_type type2 = s->type2;
            const bool             has_second = s->has_second;
            const uint64_t          seq_id = s->seq_id;
            const uint8_t * const   data1 = s->data1;
            const size_t             len1 = s->len1;
            const uint8_t * const   data2 = s->data2;
            const size_t             len2 = s->len2;
            lock.unlock();

            bool ok = pipe_send_frame(*s->socket, type1, seq_id, data1, len1);
            if (ok && has_second) {
                ok = pipe_send_frame(*s->socket, type2, seq_id, data2, len2);
            }

            lock.lock();
            s->ok       = ok;
            s->has_job  = false;
            s->job_done = true;
            lock.unlock();
            s->cv.notify_all();
        }
    }

    // Post one frame's worth of a job (non-split path).
    void post_concurrent_job(concurrent_sender & s, pipe_frame_type type, const std::vector<uint8_t> & payload,
                             uint64_t seq_id) {
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            s.type1       = type;
            s.data1       = payload.data();
            s.len1        = payload.size();
            s.has_second  = false;
            s.seq_id      = seq_id;
            s.has_job     = true;
            s.job_done    = false;
        }
        s.cv.notify_one();
    }

    // Post a two-frame job (split-frame path): BEGIN then ACTS, sent back to
    // back by the same sender thread so this socket's wire order is unchanged.
    void post_concurrent_job(concurrent_sender & s, pipe_frame_type type1, const std::vector<uint8_t> & payload1,
                             pipe_frame_type type2, const std::vector<uint8_t> & payload2, uint64_t seq_id) {
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            s.type1       = type1;
            s.data1       = payload1.data();
            s.len1        = payload1.size();
            s.type2       = type2;
            s.data2       = payload2.data();
            s.len2        = payload2.size();
            s.has_second  = true;
            s.seq_id      = seq_id;
            s.has_job     = true;
            s.job_done    = false;
        }
        s.cv.notify_one();
    }

    // Block until the job posted to this socket finishes. Returns whether the
    // send(s) succeeded; the caller (issue_requests) throws with the same
    // wording the serial path uses on failure.
    bool join_concurrent_job(concurrent_sender & s) {
        std::unique_lock<std::mutex> lock(s.mutex);
        s.cv.wait(lock, [&s]() { return s.job_done; });
        return s.ok;
    }

    void build_routes() {
        std::set<int32_t> claimed_layers;
        for (const worker & value : workers) {
            claimed_layers.insert(value.hello.layers.begin(), value.hello.layers.end());
        }
        last_routed_layer = claimed_layers.empty() ? -1 : *claimed_layers.rbegin();
        layer_slice_mode.clear();
        for (int32_t layer : claimed_layers) {
            std::vector<std::vector<size_t>> layer_routes((size_t) n_expert);
            // The workers that actually advertise THIS layer -- a mixed fleet
            // has different bands of layers claimed by disjoint worker sets
            // (e.g. slice workers for 0..42, classic CPU workers for 43..45),
            // so both the slice/classic invariants below and the mode itself
            // must be decided against this set, not against `workers` as a
            // whole.
            std::vector<const worker *> covering;
            for (const worker & value : workers) {
                if (std::find(value.hello.layers.begin(), value.hello.layers.end(), layer) !=
                    value.hello.layers.end()) {
                    covering.push_back(&value);
                }
            }
            // covering is non-empty here: `layer` came from claimed_layers,
            // which is only populated from workers' own hello.layers.
            bool all_slice   = true;
            bool all_classic = true;
            for (const worker * value : covering) {
                const bool looks_slice = value->hello.expert_first == 0 &&
                                         value->hello.expert_last == n_expert - 1 &&
                                         value->hello.shard_identity.rfind("slice:", 0) == 0;
                all_slice   = all_slice && looks_slice;
                all_classic = all_classic && !looks_slice;
            }
            if (!all_slice && !all_classic) {
                throw std::runtime_error("expert dispatcher layer " + std::to_string(layer) +
                                         " is covered by a mix of slice and classic workers");
            }
            const bool layer_is_slice = all_slice;
            layer_slice_mode.emplace(layer, layer_is_slice);

            for (int32_t expert = 0; expert < n_expert; ++expert) {
                std::set<std::string> machines;
                for (size_t i = 0; i < workers.size(); ++i) {
                    const worker & value = workers[i];
                    if (expert < value.hello.expert_first || expert > value.hello.expert_last ||
                        std::find(value.hello.layers.begin(), value.hello.layers.end(), layer) ==
                            value.hello.layers.end()) {
                        continue;
                    }
                    layer_routes[(size_t) expert].push_back(i);
                    machines.insert(value.target.machine);
                }
                if (layer_routes[(size_t) expert].empty()) {
                    throw std::runtime_error("expert dispatcher coverage gap for layer " + std::to_string(layer) +
                                             " expert " + std::to_string(expert));
                }
                if (layer_is_slice && layer_routes[(size_t) expert].size() != covering.size()) {
                    throw std::runtime_error("expert slice worker does not cover layer " + std::to_string(layer) +
                                             " expert " + std::to_string(expert));
                }
                if (!layer_is_slice && machines.size() != 1) {
                    throw std::runtime_error("expert dispatcher expert " + std::to_string(expert) + " on layer " +
                                             std::to_string(layer) + " is advertised by more than one machine");
                }
            }
            routes.emplace(layer, std::move(layer_routes));
        }
    }

    void note_in_flight_delta(dispatch_state & state, int delta) {
        if (delta > 0) {
            if (gap_at_zero) {
                deferral.ns_gap += elapsed_ns(gap_zero_since, dispatch_clock::now());
                gap_at_zero = false;
            }
            in_flight += (size_t) delta;
            if (in_flight > state.stats.max_in_flight) {
                state.stats.max_in_flight = in_flight;
            }
            std::lock_guard<std::mutex> lock(dispatch_map_mutex_);
            for (dispatch_state & slot : dispatch_slots_) {
                if (slot.handle != k_invalid_handle && in_flight > slot.stats.max_in_flight) {
                    slot.stats.max_in_flight = in_flight;
                }
            }
            return;
        }
        if (delta < 0) {
            const size_t dec = (size_t) (-delta);
            if (in_flight < dec) {
                throw std::runtime_error("expert dispatcher in-flight counter underflow");
            }
            in_flight -= dec;
            if (in_flight == 0 && !gap_at_zero) {
                gap_at_zero    = true;
                gap_zero_since = dispatch_clock::now();
            }
        }
    }

    bool is_resident(size_t worker_index, int32_t layer, int32_t expert) const {
        const worker & value = workers[worker_index];
        return value.resident_map.find(residency_key(layer, expert)) != value.resident_map.end();
    }

    int worker_port(size_t worker_index) const {
        const std::string & ep = workers[worker_index].info.endpoint;
        const size_t colon = ep.rfind(':');
        if (colon == std::string::npos) {
            return 0;
        }
        return std::atoi(ep.c_str() + colon + 1);
    }

    size_t choose_worker(int32_t                     layer,
                         int32_t                     expert,
                         const std::vector<size_t> & candidates,
                         const std::vector<size_t> & assigned_counts,
                         uint32_t                    n_tokens = 0) {
        // *** STATIC ASSIGNMENT (default ON, 2026-08-04). REPRODUCIBILITY FIX. ***
        // The balancing path below chooses from residency, in-request assigned_counts,
        // and a rotating machine_cursor -- all of which move with batch width and
        // request history. On this fleet THREE workers on THREE DIFFERENT BACKENDS
        // (CUDA 1070, Vulkan RX480, CPU) all advertise experts 0..84, so the same
        // expert could execute on a different backend from one run to the next. The
        // comment on harvest_partials already recorded the consequence: "Worker
        // ASSIGNMENT is already timing-dependent -- ~35% of requests differ between
        // identical runs". Combined with the f16 subtotals (now fixed) that silently
        // changed generated text at temperature 0 whenever the speculative draft
        // length changed.
        // Even with f32 partials a moving partition still re-associates the sum, so
        // for BITWISE reproducibility the assignment must be a PURE FUNCTION of
        // (layer, expert). A mixing hash keeps the spread without the state.
        // WP_DISPATCH_STATIC_ASSIGN=0 restores the old load-balancing behaviour --
        // faster in principle (it can prefer a worker that already holds the page)
        // but NOT reproducible. Do not turn it off for any run whose OUTPUT matters.
        // Latched per dispatcher at construction, NOT in a function-local static:
        // the prefetch-hint path asks the same question, and a process-wide
        // static would let the two answer differently for the same object.
        const bool s_static_assign = static_assign;
        if (s_static_assign && candidates.size() > 1 &&
            decode_prefer_port_ > 0 && decode_max_tokens_ > 0 &&
            n_tokens > 0 && n_tokens <= decode_max_tokens_) {
            std::vector<size_t> preferred;
            preferred.reserve(candidates.size());
            for (size_t c : candidates) {
                if (worker_port(c) == decode_prefer_port_) {
                    preferred.push_back(c);
                }
            }
            if (preferred.size() == 1) {
                return preferred[0];
            }
            if (preferred.size() > 1) {
                uint64_t h = ((uint64_t) (uint32_t) layer << 32) ^ (uint32_t) expert;
                h += 0x9E3779B97F4A7C15ull;
                h  = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
                h  = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
                h ^=  h >> 31;
                return preferred[(size_t) (h % (uint64_t) preferred.size())];
            }
        }
        if (s_static_assign && candidates.size() > 1) {
            // splitmix64 on (layer, expert): deterministic, well-spread, no state.
            uint64_t h = ((uint64_t) (uint32_t) layer << 32) ^ (uint32_t) expert;
            h += 0x9E3779B97F4A7C15ull;
            h  = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
            h  = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
            h ^=  h >> 31;
            // D6.3 weighted static assign. With default weights (all 1) this is
            // bit-for-bit the old `h % candidates.size()` pick. With a weight > 1
            // for one worker (e.g. bias the GTX 1070 over the RX 480), that
            // worker occupies that many slots in the pick table -- still a PURE
            // function of (layer, expert), so reproducibility is preserved, and
            // send_prefetch_hints calls the SAME choose_worker, so hints can
            // never disagree with dispatches.
            bool weighted = false;
            for (size_t c : candidates) {
                if (worker_weights_[c] != 1) {
                    weighted = true;
                    break;
                }
            }
            if (!weighted) {
                return candidates[(size_t) (h % (uint64_t) candidates.size())];
            }
            uint64_t total = 0;
            for (size_t c : candidates) {
                total += (uint64_t) std::max(1, worker_weights_[c]);
            }
            uint64_t pick = h % total;
            for (size_t c : candidates) {
                const uint64_t w = (uint64_t) std::max(1, worker_weights_[c]);
                if (pick < w) {
                    return c;
                }
                pick -= w;
            }
            return candidates.front();   // unreachable: total > pick by construction
        }
        if (s_static_assign) {
            return candidates.front();
        }

        bool any_resident = false;
        for (size_t candidate : candidates) {
            any_resident = any_resident || is_resident(candidate, layer, expert);
        }

        size_t              best_count      = (size_t) -1;
        uint32_t            best_slots      = 0;
        double              best_projection = 0.0;
        std::vector<size_t> tied;
        bool                use_speed = speed_split;
        if (use_speed) {
            for (size_t candidate : candidates) {
                if (any_resident && !is_resident(candidate, layer, expert)) {
                    continue;
                }
                if (workers[candidate].speed_samples < speed_estimate_min_samples ||
                    !workers[candidate].speed_fit_valid) {
                    use_speed = false;
                    break;
                }
            }
        }
        for (size_t candidate : candidates) {
            if (any_resident && !is_resident(candidate, layer, expert)) {
                continue;
            }
            const size_t   count = assigned_counts[candidate];
            const uint32_t slots = workers[candidate].hello.n_slots;
            if (!use_speed) {
                if (count < best_count || (count == best_count && slots > best_slots)) {
                    best_count = count;
                    best_slots = slots;
                    tied.clear();
                    tied.push_back(candidate);
                } else if (count == best_count && slots == best_slots) {
                    tied.push_back(candidate);
                }
                continue;
            }

            const double projection = workers[candidate].estimated_ms_per_expert * (count + 1);
            if (tied.empty() || projection < best_projection ||
                (projection == best_projection && slots > best_slots)) {
                best_projection = projection;
                best_slots      = slots;
                tied.clear();
                tied.push_back(candidate);
            } else if (projection == best_projection && slots == best_slots) {
                tied.push_back(candidate);
            }
        }

        const std::string & machine = workers[candidates.front()].target.machine;
        size_t &            cursor  = machine_cursor[machine];
        const size_t        chosen  = tied[cursor % tied.size()];
        ++cursor;
        return chosen;
    }

    void update_speed_estimate(const planned_request & request) {
        if (!speed_split || !collect_stats || request.assignments.empty() || request.wait_ns == 0) {
            return;
        }
        worker & value = workers[request.worker_index];
        value.speed_history[value.speed_history_next] = {
            request.assignments.size(), request.wait_ns * 1.0e-6,
        };
        value.speed_history_next = (value.speed_history_next + 1) % speed_estimate_window;
        if (value.speed_samples < speed_estimate_window) {
            ++value.speed_samples;
        }

        size_t min_n = (size_t) -1;
        size_t max_n = 0;
        double sum_n = 0.0;
        double sum_wait = 0.0;
        for (size_t i = 0; i < value.speed_samples; ++i) {
            const speed_sample & sample = value.speed_history[i];
            min_n = std::min(min_n, sample.n);
            max_n = std::max(max_n, sample.n);
            sum_n += (double) sample.n;
            sum_wait += sample.wait_ms;
        }
        value.speed_n_spread = max_n - min_n;
        value.speed_fit_valid = value.speed_samples >= speed_estimate_min_samples &&
                                value.speed_n_spread >= speed_estimate_min_spread;
        if (!value.speed_fit_valid) {
            return;
        }

        const double mean_n = sum_n / (double) value.speed_samples;
        const double mean_wait = sum_wait / (double) value.speed_samples;
        double       sxx = 0.0;
        double       sxy = 0.0;
        for (size_t i = 0; i < value.speed_samples; ++i) {
            const double dn = (double) value.speed_history[i].n - mean_n;
            sxx += dn * dn;
            sxy += dn * (value.speed_history[i].wait_ms - mean_wait);
        }
        const double slope = sxy / sxx;
        if (!(slope > 0.0)) {
            value.speed_fit_valid = false;
            return;
        }
        value.estimated_ms_per_expert = slope;
        value.estimated_fixed_ms = mean_wait - slope * mean_n;
    }

    void update_speed_estimates(const std::vector<planned_request> & requests) {
        for (const planned_request & request : requests) {
            update_speed_estimate(request);
        }
    }

    void log_speed_state(const std::vector<size_t> & assigned_counts) const {
        if (!speed_split || !stats_logging) {
            return;
        }
        for (size_t i = 0; i < workers.size(); ++i) {
            std::fprintf(stderr, "expert dispatch speed worker %s a=%.4f ms b=%.4f ms/expert samples=%zu n-spread=%zu assigned=%zu\n",
                         workers[i].info.endpoint.c_str(), workers[i].estimated_fixed_ms,
                         workers[i].estimated_ms_per_expert, workers[i].speed_samples,
                         workers[i].speed_n_spread, assigned_counts[i]);
        }
    }

    void update_residency(size_t worker_index, int32_t layer, const std::vector<pipe_expert_assignment> & assignments) {
        worker & value = workers[worker_index];
        for (const pipe_expert_assignment & assignment : assignments) {
            const uint64_t key = residency_key(layer, assignment.expert_id);
            auto it = value.resident_map.find(key);
            if (it != value.resident_map.end()) {
                value.resident_lru.erase(it->second);
                value.resident_map.erase(it);
            }
            value.resident_lru.emplace_back(layer, assignment.expert_id);
            value.resident_map[key] = std::prev(value.resident_lru.end());
            while (value.resident_lru.size() > value.hello.n_slots) {
                value.resident_map.erase(residency_key(value.resident_lru.front().first,
                                                       value.resident_lru.front().second));
                value.resident_lru.pop_front();
            }
        }
    }

    void log_temporal_locality() const {
        LLAMA_LOG_WARN(
            "temporal-locality: pairs=%llu mean_overlap=%.2f hist=[%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu]\n",
            (unsigned long long) temporal_n_pairs,
            (double) temporal_sum_overlap / (double) temporal_n_pairs,
            (unsigned long long) temporal_hist[0],
            (unsigned long long) temporal_hist[1],
            (unsigned long long) temporal_hist[2],
            (unsigned long long) temporal_hist[3],
            (unsigned long long) temporal_hist[4],
            (unsigned long long) temporal_hist[5],
            (unsigned long long) temporal_hist[6],
            (unsigned long long) temporal_hist[7],
            (unsigned long long) temporal_hist[8]);

        std::vector<std::pair<int32_t, const temporal_layer_stats *>> ranked;
        ranked.reserve(temporal_layers.size());
        for (const auto & entry : temporal_layers) {
            if (entry.second.n_pairs != 0) {
                ranked.emplace_back(entry.first, &entry.second);
            }
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
            const double a_mean = (double) a.second->sum_overlap / (double) a.second->n_pairs;
            const double b_mean = (double) b.second->sum_overlap / (double) b.second->n_pairs;
            return a_mean != b_mean ? a_mean < b_mean : a.first < b.first;
        });

        const size_t n = std::min((size_t) 3, ranked.size());
        for (size_t i = 0; i < n; ++i) {
            const auto & entry = ranked[i];
            LLAMA_LOG_WARN("temporal-locality: layer %d mean=%.2f pairs=%llu\n",
                           entry.first,
                           (double) entry.second->sum_overlap / (double) entry.second->n_pairs,
                           (unsigned long long) entry.second->n_pairs);
        }
        for (size_t i = 0; i < n; ++i) {
            const auto & entry = ranked[ranked.size() - 1 - i];
            LLAMA_LOG_WARN("temporal-locality: layer %d mean=%.2f pairs=%llu\n",
                           entry.first,
                           (double) entry.second->sum_overlap / (double) entry.second->n_pairs,
                           (unsigned long long) entry.second->n_pairs);
        }
    }

    void add_temporal_pair(temporal_layer_stats & layer_stats, size_t overlap) {
        ++layer_stats.n_pairs;
        layer_stats.sum_overlap += overlap;
        ++layer_stats.hist[std::min(overlap, layer_stats.hist.size() - 1)];
        ++temporal_n_pairs;
        temporal_sum_overlap += overlap;
        ++temporal_hist[std::min(overlap, temporal_hist.size() - 1)];
        if (temporal_n_pairs % 4096 == 0) {
            log_temporal_locality();
        }
    }

    void add_temporal_locality(int32_t                                     layer,
                               uint32_t                                    n_tokens,
                               const std::vector<pipe_expert_assignment> & assignments) {
        if (!s_temporal_stats) {
            return;
        }

        temporal_layer_stats & layer_stats = temporal_layers[layer];
        if (n_tokens > 16) {
            layer_stats.previous_experts.clear();
            return;
        }

        std::vector<int32_t> experts;
        experts.reserve(std::min(assignments.size(), (size_t) std::max(n_expert_used, 0)));
        for (uint32_t token = 0; token < n_tokens; ++token) {
            experts.clear();
            for (const pipe_expert_assignment & assignment : assignments) {
                if (assignment.weights[(size_t) token] != 0.0f) {
                    experts.push_back(assignment.expert_id);
                }
            }
            std::sort(experts.begin(), experts.end());

            if (!layer_stats.previous_experts.empty()) {
                size_t overlap = 0;
                size_t previous = 0;
                size_t current = 0;
                while (previous < layer_stats.previous_experts.size() && current < experts.size()) {
                    const int32_t previous_expert = layer_stats.previous_experts[previous];
                    const int32_t current_expert = experts[current];
                    if (previous_expert == current_expert) {
                        ++overlap;
                        ++previous;
                        ++current;
                    } else if (previous_expert < current_expert) {
                        ++previous;
                    } else {
                        ++current;
                    }
                }
                add_temporal_pair(layer_stats, overlap);
            }

            // This also links the first token of a later dispatch. Across
            // accept/reject boundaries it is only an approximate decode order.
            layer_stats.previous_experts = experts;
        }
    }

    void dump_routing(int32_t                                     layer,
                      uint32_t                                    n_tokens,
                      const std::vector<pipe_expert_assignment> & assignments) {
        if (routing_dump_ == nullptr || n_tokens > 8) {
            return;
        }
        if (routing_dump_last_layer_ >= 0 && layer < routing_dump_last_layer_) {
            ++routing_dump_step_;
        }
        routing_dump_last_layer_ = layer;

        std::vector<int32_t> experts;
        experts.reserve(assignments.size());
        std::fprintf(routing_dump_, "B %llu %d %u", (unsigned long long) routing_dump_step_, layer, n_tokens);
        for (uint32_t token = 0; token < n_tokens; ++token) {
            experts.clear();
            for (const pipe_expert_assignment & assignment : assignments) {
                if (assignment.weights[(size_t) token] != 0.0f) {
                    experts.push_back(assignment.expert_id);
                }
            }
            std::sort(experts.begin(), experts.end());
            std::fputs(" |", routing_dump_);
            for (int32_t expert : experts) {
                std::fprintf(routing_dump_, " %d", expert);
            }
        }
        std::fputc('\n', routing_dump_);
        std::fflush(routing_dump_);
    }

    void poison() {
        poisoned  = true;
        in_flight = 0;
        gap_at_zero = false;
        pending_def = {};
        {
            std::lock_guard<std::mutex> lock(dispatch_map_mutex_);
            dispatch_slots_ = {};
        }
        // D1 async: stop and join the writer threads first, then drop sockets.
        // A writer blocked in send() unblocks when the peer's connection breaks
        // (the reason we are poisoning) or once it sees the stop flag; joining
        // here guarantees no thread is left running into impl storage. Idempotent
        // with ~impl's stop_writers() -- the threads are already joined by the
        // time the object is destroyed. Same reasoning for the concurrent-issue
        // senders.
        stop_writers(true);
        stop_concurrent_senders();
        for (worker & value : workers) {
            value.socket.reset();
        }
    }

    // Spin on a zero-timeout poll() until this socket has readable bytes or the
    // WP_AWAIT_SPIN_US budget is exhausted. No-op when the knob is unset. Never
    // consumes anything from the socket -- the caller's recv path is unchanged
    // whether this returns because data arrived or because it gave up.
    void spin_for_readable(pipe_socket_t & sock) {
        if (s_await_spin_us == 0) {
            return;
        }
        const int fd = sock.poll_fd();
        if (fd < 0) {
            return;
        }
        const auto deadline = dispatch_clock::now() + std::chrono::microseconds(s_await_spin_us);
        struct pollfd pfd;
        pfd.fd      = fd;
        pfd.events  = POLLIN;
        do {
            pfd.revents = 0;
            const int r = ::poll(&pfd, 1, 0);
            if (r != 0) {
                // readable, or an error the blocking recv will surface properly
                return;
            }
        } while (dispatch_clock::now() < deadline);
    }

    pipe_frame_type await_response(planned_request & request, uint64_t wanted_seq_id,
                                   std::vector<uint8_t> & payload, dispatch_state & state) {
        if (!state.stats.first_await_recorded) {
            state.stats.first_await_recorded  = true;
            state.stats.first_await_in_flight = in_flight;
        }
        pipe_frame_type type;
        uint64_t        seq_id = 0;
        worker &        value  = workers[request.worker_index];
        // D1 async: if this worker's writer thread already failed (its send did
        // not complete), the partial we are about to await will never arrive.
        // Surface that now instead of blocking on a dead socket.
        if (value.inproc) {
            const pipe_expert_partial partial = value.inproc->dispatch(request.inproc_wire);
            payload = pipe_encode_expert_partial(partial);
            return PIPE_EXPERT_PARTIAL;
        }
        if (async_issue) {
            socket_writer * w = value.writer.get();
            if (w) {
                std::lock_guard<std::mutex> lock(w->mutex);
                if (w->failed) {
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " writer failed: " + w->error_msg);
                }
            }
        }
        // WP_AWAIT_SPIN_US: burn the tail of the wait on-core so the response
        // is consumed without a kernel wakeup. Purely a timing change; on
        // timeout, on error, or when disabled we fall into the identical
        // blocking pipe_recv_frame below.
        spin_for_readable(*value.socket);
        if (!pipe_recv_frame(*value.socket, type, seq_id, payload)) {
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                     " died while computing expert(s) " + assignment_experts(request.assignments));
        }
        // The throw condition is EXACTLY the original strict compare. It must
        // not also require demux_seq_id() to resolve: deferred-request
        // responses (WP_DEFER_K) are legitimately awaited AFTER their
        // dispatch's slot is released, and gates49 (2026-08-31) showed even
        // the immediate path awaits after release depending on call order --
        // gating the throw on the slot map made every await fail with
        // "returned sequence N while awaiting N". demux_seq_id() stays as the
        // routing lookup stage 4 will use to steer a response to one of two
        // open handles; until then it is not a validity check.
        if (seq_id != wanted_seq_id) {
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint + " returned sequence " +
                                     std::to_string(seq_id) + " while awaiting " + std::to_string(wanted_seq_id));
        }
        return type;
    }

    std::vector<planned_request> plan_requests(int32_t                                     layer,
                                               uint32_t                                    n_tokens,
                                               const std::vector<float> &                  activations,
                                               const std::vector<pipe_expert_assignment> & assignments,
                                               const std::vector<std::vector<size_t>> &   layer_routes,
                                               std::vector<size_t> &                       assigned_counts,
                                               float                                       swiglu_clamp) {
        std::vector<planned_request> by_worker(workers.size());
        for (size_t i = 0; i < workers.size(); ++i) {
            by_worker[i].worker_index = i;
            by_worker[i].layer = layer;
        }
        const bool layer_is_slice = layer_slice_mode.at(layer);
        for (const pipe_expert_assignment & assignment : assignments) {
            const std::vector<size_t> & candidates = layer_routes[(size_t) assignment.expert_id];
            if (layer_is_slice) {
                for (size_t worker_index : candidates) {
                    by_worker[worker_index].assignments.push_back(assignment);
                    ++assigned_counts[worker_index];
                }
                continue;
            }
            const size_t chosen = choose_worker(layer, assignment.expert_id, candidates, assigned_counts, n_tokens);
            by_worker[chosen].assignments.push_back(assignment);
            ++assigned_counts[chosen];
        }

        std::vector<planned_request> requests;
        // WP_SLICE_ENCODE_ONCE: on a sliced layer the covering workers' frames
        // are byte-identical, so build + encode once (first covering worker) and
        // reuse the bytes for the rest. Disabled under WP_DISPATCH_UNION so its
        // per-worker diagnostic still runs, and only for n_tokens>1 (the encode
        // cost this targets is a prefill cost).
        const bool slice_encode_once =
            layer_is_slice && s_slice_encode_once && !s_union_stats && n_tokens > 1;
        // See planned_request::split_wire. Split the frame only at a width where
        // dedup can actually engage; below it the second frame buys nothing and
        // costs a packet per request on a TCP_NODELAY socket. Uniform across a
        // layer's workers (n_tokens is a property of the ubatch), so the
        // slice_encode_once shared-payload cache below stays coherent.
        const bool split_wire = split_frame && n_tokens > dedup_min_tokens_;
        std::vector<uint8_t>  shared_payload, shared_begin_payload, shared_acts_payload;
        std::vector<uint32_t> shared_token_ids;
        bool have_shared = false;
        for (planned_request & request : by_worker) {
            if (request.assignments.empty()) {
                continue;
            }
            if (slice_encode_once && have_shared) {
                // Identical-frame fast path: copy the already-encoded bytes.
                request.token_ids = shared_token_ids;
                request.split_wire = split_wire;
                if (split_wire) {
                    request.begin_payload = shared_begin_payload;
                    request.acts_payload  = shared_acts_payload;
                } else {
                    request.payload = shared_payload;
                }
                requests.push_back(std::move(request));
                continue;
            }
            // *** WP_DISPATCH_UNION=1: MEASUREMENT ONLY, no behaviour change. ***
            // Every worker currently receives the FULL activation tensor
            // [n_embd x n_tokens], but it only needs the token rows that route to
            // ITS assigned experts -- the workers already exploit that internally
            // via the gather path, AFTER paying to receive everything. At prefill
            // that is ~33.5 MB per request (2048 x 4096 x f32) and `issue` is 8.2 s
            // of a 26.4 s prefill dispatch, so the send path is on the critical
            // path, not incidental.
            // This logs the UNION of tokens with a nonzero routing weight across a
            // worker's assignments -- NOT the token-expert pair count, which
            // n_weight_nonzero already reports and which overcounts a token that
            // hits several of the same worker's experts. |union| / n_tokens is the
            // exact factor a spine-side gather would shrink the payload by.
            std::vector<uint32_t> needed;
            // WP_SLICE_SKIP_SCAN: skip this union scan on a sliced layer -- it
            // always yields needed.size()==n_tokens there (compact branch never
            // taken), so it is pure spine CPU. Kept when WP_DISPATCH_UNION is
            // measuring so its per-worker log still runs.
            if ((s_gather || s_union_stats) && n_tokens > 1 &&
                !(layer_is_slice && s_slice_skip_scan && !s_union_stats)) {
                std::vector<uint8_t> touched((size_t) n_tokens, 0);
                for (const pipe_expert_assignment & a : request.assignments) {
                    for (uint32_t t = 0; t < n_tokens; ++t) {
                        if (a.weights[(size_t) t] != 0.0f) {
                            touched[(size_t) t] = 1;
                        }
                    }
                }
                for (uint32_t t = 0; t < n_tokens; ++t) {
                    if (touched[(size_t) t]) {
                        needed.push_back(t);
                    }
                }
                if (s_union_stats) {
                    std::fprintf(stderr,
                        "expert dispatch union: layer=%d worker=%zu n_tokens=%u experts=%zu "
                        "tokens_needed=%zu (%.2f%%) bytes_sent=%zu bytes_needed=%zu\n",
                        layer, request.worker_index, n_tokens, request.assignments.size(),
                        needed.size(), 100.0 * (double) needed.size() / (double) n_tokens,
                        (size_t) n_tokens * (size_t) n_embd * sizeof(float),
                        needed.size() * (size_t) n_embd * sizeof(float));
                }
            }

            pipe_expert_dispatch_req wire_request;
            wire_request.layer        = layer;
            wire_request.swiglu_clamp = swiglu_clamp;
            // *** SPINE-SIDE GATHER: send only the rows this worker needs. ***
            // Skipped when it needs every row, so the loopback worker (99.8%) and
            // every decode step take the identity path and stay bit-identical.
            // request.assignments keeps the FULL weights on purpose -- it is what
            // the error messages and per-request logs read expert ids from; only
            // the WIRE copy is compacted.
            // *** REQUIRE A MEANINGFUL SAVING, NOT MERELY A NONZERO ONE. ***
            // "needed < n_tokens" alone fires for a worker that needs 658 of 659
            // rows, which is exactly the R9700: it holds 130 of 256 experts, so
            // measured union is 99.8%. It would gather to drop ONE row -- and the
            // non-gather path copies the full activation anyway, so nothing is
            // saved on the copy. What it costs is real: compacting 130 assignment
            // weight vectors, building a 658-entry index, and taking scatter_add's
            // row-by-row accumulate instead of the flat elementwise add over 658
            // of 659 rows. All to avoid 16 KB on LOOPBACK, where bytes are free.
            // The two workers this is for sit at 68.0% / 67.9%, far inside the
            // threshold; the fraction is structural, 1-(1-E/256)^8, so a worker
            // only clears it below roughly E=110 experts.
            const size_t gather_max_rows =
                (size_t) ((double) n_tokens * s_gather_max_frac);
            if (s_gather && !needed.empty() && needed.size() <= gather_max_rows) {
                const size_t rows = needed.size();
                wire_request.n_tokens = (uint32_t) rows;
                wire_request.assignments.reserve(request.assignments.size());
                for (const pipe_expert_assignment & a : request.assignments) {
                    pipe_expert_assignment compact;
                    compact.expert_id = a.expert_id;
                    compact.weights.resize(rows);
                    for (size_t r = 0; r < rows; ++r) {
                        compact.weights[r] = a.weights[(size_t) needed[r]];
                    }
                    wire_request.assignments.push_back(std::move(compact));
                }
                wire_request.activations.resize(rows * (size_t) n_embd);
                for (size_t r = 0; r < rows; ++r) {
                    const float * src = activations.data() + (size_t) needed[r] * (size_t) n_embd;
                    std::copy(src, src + n_embd,
                              wire_request.activations.begin() + (ptrdiff_t) (r * (size_t) n_embd));
                }
                request.token_ids = std::move(needed);
            } else {
                wire_request.n_tokens    = n_tokens;
                wire_request.assignments = request.assignments;
                wire_request.activations = activations;
            }
            if (workers[request.worker_index].inproc) {
                request.inproc_wire = std::move(wire_request);
                requests.push_back(std::move(request));
                continue;
            }
            const dispatch_clock::time_point encode_started =
                layer_trace_enabled() ? dispatch_clock::now() : dispatch_clock::time_point{};
            request.split_wire = split_wire;
            if (split_wire) {
                pipe_expert_dispatch_begin begin;
                begin.layer = wire_request.layer;
                begin.n_tokens = wire_request.n_tokens;
                begin.assignments = wire_request.assignments;
                begin.swiglu_clamp = wire_request.swiglu_clamp;
                pipe_expert_dispatch_acts acts;
                acts.activations = wire_request.activations;
                request.begin_payload = pipe_encode_expert_dispatch_begin(begin);
                request.acts_payload = pipe_encode_expert_dispatch_acts(acts);
            } else {
                request.payload = pipe_encode_expert_dispatch_req(wire_request);
            }
            if (layer_trace_enabled()) {
                add_layer_trace(layer, &layer_trace_stats::encode_ns,
                                elapsed_ns(encode_started, dispatch_clock::now()));
            }
            if (slice_encode_once) {
                // Stash the first covering worker's encoded frame for reuse.
                shared_token_ids = request.token_ids;
                if (split_wire) {
                    shared_begin_payload = request.begin_payload;
                    shared_acts_payload  = request.acts_payload;
                } else {
                    shared_payload = request.payload;
                }
                have_shared = true;
            }
            requests.push_back(std::move(request));
        }
        return requests;
    }

    // Partition `experts` across the workers that will actually be asked for
    // them and send one hint frame each. See dispatcher::send_prefetch_hints.
    size_t send_prefetch_hints(int32_t layer, const std::vector<int32_t> & experts,
                               uint32_t provenance, uint32_t n_tokens) {
        if (poisoned || experts.empty()) {
            return 0;
        }
        // The old default protects the request/response stream conservatively.
        // The worker receives complete frames and dispatches hints by type, so
        // WP_HINT_INFLIGHT=1 can safely use the same socket while a response is
        // outstanding.
        const bool sent_in_flight = in_flight != 0;
        if (sent_in_flight && !hint_inflight) {
            ++hint_stats.n_skipped_in_flight;
            return 0;
        }
        const auto route = routes.find(layer);
        if (route == routes.end()) {
            ++hint_stats.n_no_oracle;
            return 0;
        }
        // WHY THIS DECLINES INSTEAD OF GUESSING. Under WP_DISPATCH_STATIC_ASSIGN
        // (default on) choose_worker is a PURE FUNCTION of (layer, expert,
        // n_tokens) -- a splitmix64 hash, plus the decode-prefer port filter
        // when 0 < n_tokens <= WP_DISPATCH_DECODE_MAX_TOKENS -- so the hint can
        // name the exact worker the dispatch will use. Pass the SAME n_tokens
        // the request will use; n_tokens=0 skips the prefer filter. With it off, the choice moves with residency,
        // in-request counts and a rotating machine cursor, none of which exist
        // yet at hint time. A wrong guess is not neutral: on this fleet the
        // 1070, the RX 480 and the CPU worker all advertise experts 0..84 and
        // the first two READ THE SAME SHARD OFF THE SAME DRIVE, so hinting the
        // wrong one buys a wasted read AND leaves the real one cold. That is
        // precisely the wasted read that killed the 2026-07 attempt.
        if (!static_assign) {
            ++hint_stats.n_skipped_dynamic;
            return 0;
        }

        const std::vector<std::vector<size_t>> & layer_routes = route->second;
        // assigned_counts is unused under static assignment; pass a zero vector
        // so the one call site stays identical to the dispatch path's.
        const std::vector<size_t> no_counts(workers.size(), 0);
        const bool                layer_is_slice = layer_slice_mode.at(layer);

        std::vector<std::vector<int32_t>> by_worker(workers.size());
        for (int32_t expert : experts) {
            if (expert < 0 || expert >= n_expert) {
                continue;
            }
            const std::vector<size_t> & candidates = layer_routes[(size_t) expert];
            if (candidates.empty()) {
                continue;
            }
            if (layer_is_slice) {
                for (size_t worker_index : candidates) {
                    by_worker[worker_index].push_back(expert);
                }
            } else {
                by_worker[choose_worker(layer, expert, candidates, no_counts, n_tokens)].push_back(expert);
            }
        }

        size_t sent = 0;
        for (size_t i = 0; i < by_worker.size(); ++i) {
            if (by_worker[i].empty()) {
                continue;
            }
            worker & value = workers[i];
            if (!value.socket) {
                continue;
            }
            pipe_expert_prefetch_hint hint;
            hint.layer      = layer;
            hint.provenance = provenance;
            hint.expert_ids = std::move(by_worker[i]);   // already ascending: `experts` is
            const std::vector<uint8_t> payload = pipe_encode_expert_prefetch_hint(hint);
            // seq_id 0: a hint is never correlated with a response, so it must not
            // consume an id from the request sequence.
            if (async_issue) {
                // D1: hints go through the same per-socket FIFO writer queue as
                // requests. Two threads calling send() on one socket could
                // interleave bytes inside a frame -- routing everything through
                // the single writer thread makes that impossible, and the FIFO
                // preserves the wire-order invariant (a hint can never overtake a
                // request frame on the same socket).
                wire_frame frame;
                frame.type    = PIPE_EXPERT_PREFETCH_HINT;
                frame.seq_id  = 0;
                frame.payload = payload;   // small; copy is fine
                try {
                    enqueue_frame(value, std::move(frame));
                } catch (...) {
                    ++hint_stats.n_send_failed;
                    continue;
                }
            } else if (!pipe_send_frame(*value.socket, PIPE_EXPERT_PREFETCH_HINT, 0,
                                        payload.data(), payload.size())) {
                ++hint_stats.n_send_failed;
                continue;
            }
            ++hint_stats.n_frames;
            hint_stats.n_experts += (uint64_t) hint.expert_ids.size();
            if (sent_in_flight) {
                ++hint_stats.n_sent_in_flight;
            }
            ++sent;
        }
        return sent;
    }

    // WP_DISPATCH_DEDUP_ACTIVATIONS. `requests` is plan_requests()'s IMMEDIATE
    // output for one layer -- one entry per worker with a non-empty
    // assignment. Group requests by machine; for every machine with >= 2
    // members here, elect the lowest worker_index as PRIMARY and send it the
    // full activations with a publish request. Synchronously await that
    // primary's publish ack (this call blocks on the primary's socket) BEFORE
    // sending anything to the rest of the group ("secondaries"):
    //
    //   ack.success == true  -> secondaries get PIPE_EXPERT_DISPATCH_ACTS_REF
    //                           (no activation bytes; they read the shm the
    //                           primary just published).
    //   ack.success == false -> secondaries get the ORDINARY full ACTS, i.e.
    //                           today's path, exactly as if dedup had never
    //                           been attempted for this group. Zero secondary-
    //                           side risk: nothing was ever promised to them.
    //
    // WHY THE ACK MUST BE AWAITED BEFORE THE REF IS SENT (this is the whole
    // safety argument, not an optimisation detail). The primary sets its shm
    // segment's ready flag and unlinks-on-last-out using a refcount seeded at
    // publish time; it can only send PIPE_EXPERT_ACTS_PUBLISH_ACK AFTER that
    // segment exists and is marked ready (see the ACTS_PUBLISH handler in
    // wp-expert-worker.cpp). The spine's own program order then makes it
    // impossible to send a REF before that ack has been read on THIS thread --
    // there is no race to win, because there is nothing to race: the REF is
    // simply never sent until the fact it depends on is already true. A
    // secondary can therefore only ever open a segment that has already been
    // created and marked ready by the time the REF that names it exists. The
    // remaining failure mode (secondary-side shm_open/mmap failure despite the
    // segment being real and ready -- a purely local fault: permissions,
    // memory pressure, a kernel shm limit) is handled by the retry in
    // receive_partial(), not here.
    //
    // COST OF THIS ORDERING: the primary's publish-ack round trip is
    // synchronous on the dispatch thread, so secondaries in a dedup group are
    // issued strictly after the primary's ack, not concurrently with it. That
    // ack is small (2 bytes of payload) and local-ish (one 1 GbE round trip to
    // the SAME machine the bytes were already being sent to), which is why
    // this is gated to n_tokens > WP_DISPATCH_DEDUP_MIN_TOKENS -- at that
    // width the ack's cost is noise next to the multi-hundred-KB-to-multi-MB
    // per-worker payload it is replacing.
    //
    // BIT-EXACTNESS. layer_is_slice callers only: in slice mode every covering
    // worker's wire activations are the FULL, un-gathered [n_tokens x n_embd]
    // tensor (WP_DISPATCH_UNION measured tokens_needed==100% for exactly this
    // reason -- see the PIPE_VERSION 14 comment in pipe-protocol.h), so the
    // primary and every secondary in a machine group are computing on
    // byte-identical activations by construction: the secondary's copy IS the
    // primary's copy, read out of the same shm bytes the primary wrote, not a
    // re-encode. request.token_ids must be empty for every group member or
    // this function declines the whole group (defensive; should be
    // unreachable for a slice-mode layer under WP_DISPATCH_GATHER, which never
    // fires there, but this is not the place to assume that silently).
    void dedup_publish_and_ref(std::vector<planned_request> & requests, uint64_t seq_id,
                               int32_t layer, uint32_t n_tokens,
                               const std::vector<float> & activations, dispatch_state & state) {
        if (!dedup_activations || n_tokens <= dedup_min_tokens_ || !layer_slice_mode.at(layer)) {
            return;
        }
        std::map<std::string, std::vector<size_t>> groups; // machine -> indices into `requests`
        for (size_t i = 0; i < requests.size(); ++i) {
            if (!requests[i].token_ids.empty()) {
                // A gathered (compacted) request on a layer this function
                // otherwise treats as slice-mode-identical. Should not happen
                // (see the comment above) but bit-exactness is not something
                // to gamble on: skip dedup for this request's worker entirely
                // by simply never adding it to a group.
                continue;
            }
            groups[workers[requests[i].worker_index].info.machine].push_back(i);
        }
        for (auto & [machine, idxs] : groups) {
            if (idxs.size() < 2) {
                continue; // nothing co-located this round; ordinary path.
            }
            std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
                return requests[a].worker_index < requests[b].worker_index;
            });
            const size_t primary_i = idxs.front();
            planned_request & primary = requests[primary_i];
            worker &          primary_worker = workers[primary.worker_index];

            pipe_expert_dispatch_acts_publish publish;
            publish.n_subscribers = (uint32_t) (idxs.size() - 1);
            publish.activations   = activations; // one copy; see below for why.
            const std::vector<uint8_t> publish_payload = pipe_encode_expert_dispatch_acts_publish(publish);

            if (!pipe_send_frame(*primary_worker.socket, PIPE_EXPERT_DISPATCH_BEGIN, seq_id,
                                 primary.begin_payload.data(), primary.begin_payload.size()) ||
                !pipe_send_frame(*primary_worker.socket, PIPE_EXPERT_DISPATCH_ACTS_PUBLISH, seq_id,
                                 publish_payload.data(), publish_payload.size())) {
                throw std::runtime_error("expert dispatcher failed to send dedup publish to worker " +
                                         primary_worker.info.endpoint);
            }
            note_in_flight_delta(state, +1);
            ++state.stats.requests_issued;
            primary.already_issued = true;

            // Synchronous: block on the primary's own ack before touching any
            // secondary. This frame is NOT the primary's PIPE_EXPERT_PARTIAL --
            // that is awaited later, normally, through the regular harvest
            // path, exactly like every other request in `requests`.
            pipe_frame_type      ack_type;
            uint64_t             ack_seq = 0;
            std::vector<uint8_t> ack_payload;
            if (!pipe_recv_frame(*primary_worker.socket, ack_type, ack_seq, ack_payload)) {
                throw std::runtime_error("expert dispatcher worker " + primary_worker.info.endpoint +
                                         " died while publishing dedup activations");
            }
            if (ack_seq != seq_id || ack_type != PIPE_EXPERT_ACTS_PUBLISH_ACK) {
                throw std::runtime_error("expert dispatcher worker " + primary_worker.info.endpoint +
                                         " sent an unexpected frame in place of the dedup publish ack");
            }
            const pipe_expert_acts_publish_ack ack =
                pipe_decode_expert_acts_publish_ack(ack_payload.data(), ack_payload.size());

            // Built once regardless of ack.success: needed as the fallback
            // payload on every secondary either way (immediately, if the
            // publish failed; later from receive_partial, if a secondary's
            // own shm read fails despite success). Same bytes primary just
            // published -- pipe_encode_expert_dispatch_acts encodes the exact
            // same `activations` vector, not a re-derivation.
            pipe_expert_dispatch_acts plain;
            plain.activations = activations;
            const std::vector<uint8_t> fallback_payload = pipe_encode_expert_dispatch_acts(plain);

            pipe_expert_dispatch_acts_ref ref;
            ref.n_tokens = n_tokens;
            const std::vector<uint8_t> ref_payload = pipe_encode_expert_dispatch_acts_ref(ref);

            for (size_t k = 1; k < idxs.size(); ++k) {
                planned_request & secondary = requests[idxs[k]];
                worker &          secondary_worker = workers[secondary.worker_index];
                secondary.dedup_role_secondary       = ack.success;
                secondary.dedup_fallback_acts_payload = fallback_payload;
                const bool sent =
                    pipe_send_frame(*secondary_worker.socket, PIPE_EXPERT_DISPATCH_BEGIN, seq_id,
                                    secondary.begin_payload.data(), secondary.begin_payload.size()) &&
                    (ack.success
                         ? pipe_send_frame(*secondary_worker.socket, PIPE_EXPERT_DISPATCH_ACTS_REF, seq_id,
                                           ref_payload.data(), ref_payload.size())
                         : pipe_send_frame(*secondary_worker.socket, PIPE_EXPERT_DISPATCH_ACTS, seq_id,
                                           fallback_payload.data(), fallback_payload.size()));
                if (!sent) {
                    throw std::runtime_error("expert dispatcher failed to send dedup request to worker " +
                                             secondary_worker.info.endpoint);
                }
                note_in_flight_delta(state, +1);
                ++state.stats.requests_issued;
                secondary.already_issued = true;
            }
        }
    }

    void issue_requests(std::vector<planned_request> & requests, uint64_t seq_id, dispatch_state & state) {
        // WP_ISSUE_WIDEST_FIRST: walk `requests` in issue_order (widest slice
        // first) so the long-pole worker's bytes hit the wire earliest. requests
        // is not necessarily one-per-worker (a non-slice layer skips workers with
        // no assignment), so index by worker and skip absent entries. When the
        // flag is off, issue_order is 0,1,2,... and this reproduces the original
        // order exactly. Fold order is unaffected (harvest uses requests order).
        std::vector<planned_request *> by_worker_index(workers.size(), nullptr);
        for (planned_request & request : requests) {
            by_worker_index[request.worker_index] = &request;
        }
        // WP_CONCURRENT_ISSUE: requests posted to a socket's sender thread in
        // this pass are joined in this same fixed (issue_order) order in the
        // second pass below, once every socket's job has been posted -- so the
        // wall time this loop pays for sends collapses to the MAX per-link send
        // instead of their SUM, while still throwing on the first FAILED send
        // in the same order the serial path would have reached it.
        std::vector<std::pair<worker *, planned_request *>> concurrent_pending;
        for (size_t widx : issue_order) {
            planned_request * req_ptr = by_worker_index[widx];
            if (req_ptr == nullptr) {
                continue;
            }
            planned_request & request = *req_ptr;
            if (request.already_issued) {
                // WP_DISPATCH_DEDUP_ACTIVATIONS already put this request's
                // BEGIN+ACTS(-variant) on the wire synchronously in
                // dedup_publish_and_ref() (a secondary's ACTS_REF cannot be
                // sent before its primary's publish is acknowledged, so it
                // cannot go through this generic per-worker loop). in_flight
                // and stats.requests_issued were already updated there too.
                continue;
            }
            worker & value = workers[request.worker_index];
            if (value.inproc) {
                // Do NOT HIP-compute here. Issue runs inside the spine's
                // graph_compute while WP_HIP_GRAPHS may be capturing the next
                // GPU split; hipMemcpy from a second backend then aborts
                // ("legacy stream depend on a capturing blocking stream").
                // Wait/finish_dispatch is after that capture window.
                if (collect_stats || req_log_ != nullptr) {
                    request.issued_at = dispatch_clock::now();
                }
                note_in_flight_delta(state, +1);
                ++state.stats.requests_issued;
                continue;
            }
            const auto send_frame = [&](pipe_frame_type type, const std::vector<uint8_t> & payload) {
                if (!layer_trace_enabled()) {
                    return pipe_send_frame(*value.socket, type, seq_id, payload.data(), payload.size());
                }
                const dispatch_clock::time_point send_started = dispatch_clock::now();
                const bool send_ok = pipe_send_frame(*value.socket, type, seq_id, payload.data(), payload.size());
                add_layer_trace(request.layer, &layer_trace_stats::send_ns,
                                elapsed_ns(send_started, dispatch_clock::now()));
                return send_ok;
            };
            if (collect_stats || req_log_ != nullptr) {
                request.issued_at = dispatch_clock::now();
            }
            if (request.split_wire && async_issue) {
                wire_frame begin;
                begin.type = PIPE_EXPERT_DISPATCH_BEGIN;
                begin.seq_id = seq_id;
                begin.layer = request.layer;
                begin.payload = std::move(request.begin_payload);
                enqueue_frame(value, std::move(begin));
                wire_frame acts;
                acts.type = PIPE_EXPERT_DISPATCH_ACTS;
                acts.seq_id = seq_id;
                acts.layer = request.layer;
                acts.payload = std::move(request.acts_payload);
                enqueue_frame(value, std::move(acts));
            } else if (request.split_wire && concurrent_issue) {
                // Scatter: post BEGIN+ACTS to this worker's persistent sender
                // thread and move on to the next worker immediately -- do NOT
                // block here. All sockets' sends run concurrently; joined below
                // once every worker in this dispatch has been posted.
                post_concurrent_job(*value.sender, PIPE_EXPERT_DISPATCH_BEGIN, request.begin_payload,
                                    PIPE_EXPERT_DISPATCH_ACTS, request.acts_payload, seq_id);
                concurrent_pending.emplace_back(&value, &request);
                continue;
            } else if (request.split_wire) {
                if (!send_frame(PIPE_EXPERT_DISPATCH_BEGIN, request.begin_payload) ||
                    !send_frame(PIPE_EXPERT_DISPATCH_ACTS, request.acts_payload)) {
                    throw std::runtime_error("expert dispatcher failed to send split expert request to worker " +
                                             value.info.endpoint);
                }
            } else if (async_issue) {
                // D1: enqueue and return. The per-socket writer thread moves the
                // bytes; the dispatch thread is free to issue the next worker /
                // layer immediately. payload is moved out (bytes only) -- the
                // request keeps its assignments/token_ids for the await path.
                wire_frame frame;
                frame.type    = PIPE_EXPERT_DISPATCH_REQ;
                frame.seq_id  = seq_id;
                frame.layer   = request.layer;
                frame.payload = std::move(request.payload);
                enqueue_frame(value, std::move(frame));
            } else if (concurrent_issue) {
                // Scatter (non-split path); see the split_frame branch above.
                post_concurrent_job(*value.sender, PIPE_EXPERT_DISPATCH_REQ, request.payload, seq_id);
                concurrent_pending.emplace_back(&value, &request);
                continue;
            } else if (!send_frame(PIPE_EXPERT_DISPATCH_REQ, request.payload)) {
                throw std::runtime_error("expert dispatcher failed to send expert(s) " +
                                         assignment_experts(request.assignments) + " to worker " +
                                         value.info.endpoint);
            }
            note_in_flight_delta(state, +1);
            ++state.stats.requests_issued;
        }
        // Join pass: block on each posted job in the same fixed order the
        // requests were posted (== issue_order), so a failure surfaces against
        // the same worker the serial path would have failed on first. This is
        // where the wait for "slowest link" actually happens -- everything
        // before this point only enqueued a job.
        for (auto & [worker_ptr, request_ptr] : concurrent_pending) {
            if (!join_concurrent_job(*worker_ptr->sender)) {
                if (request_ptr->split_wire) {
                    throw std::runtime_error("expert dispatcher failed to send split expert request to worker " +
                                             worker_ptr->info.endpoint);
                }
                throw std::runtime_error("expert dispatcher failed to send expert(s) " +
                                         assignment_experts(request_ptr->assignments) + " to worker " +
                                         worker_ptr->info.endpoint);
            }
            note_in_flight_delta(state, +1);
            ++state.stats.requests_issued;
        }
    }

    // Receive ONE partial and decode it into `out` (does NOT accumulate). Split
    // out of accumulate_partial so the caller can harvest partials in ARRIVAL
    // order while still summing them in a FIXED order -- see harvest_partials.
    void write_request_log(const planned_request & request, int32_t layer, uint32_t n_tokens,
                           const dispatch_state & state) {
        if (req_log_ == nullptr) {
            return;
        }
        // Column order: layer n_tokens worker_index n_experts ns_before_await
        // ns_blocked ns_issue_done ns_await_recv resp_bytes ns_unpack
        // await_start_ns await_end_ns seq_id chunk_index.
        fprintf(req_log_, "%d %u %zu %zu %llu %llu %llu %llu %llu %llu %llu %llu %llu %u\n",
                layer, n_tokens, request.worker_index, request.assignments.size(),
                (unsigned long long) elapsed_ns(request.issued_at, request.await_started_at),
                (unsigned long long) elapsed_ns(request.await_started_at, request.await_finished_at),
                (unsigned long long) elapsed_ns(request.issued_at, request.await_started_at),
                (unsigned long long) elapsed_ns(request.await_started_at, request.await_finished_at),
                (unsigned long long) request.response_bytes,
                (unsigned long long) request.unpack_ns,
                (unsigned long long) elapsed_ns(state.req_dispatch_start_, request.await_started_at),
                (unsigned long long) elapsed_ns(state.req_dispatch_start_, request.await_finished_at),
                (unsigned long long) state.seq_id, state.chunk_index);
        fflush(req_log_);
    }

    void receive_partial(std::vector<float> &             out,
                         size_t                           n_values,
                         planned_request &                request,
                         uint64_t                         seq_id,
                         int32_t                          layer,
                         uint32_t                         n_tokens,
                         dispatch_clock::time_point *     last_response,
                         dispatch_state &                 state) {
        std::vector<uint8_t>  payload;
        // WP_DISPATCH_REQ_LOG=path: one line per request. The complete column
        // order is documented at write_request_log below.
        //
        // n_tokens added 2026-08-03: prefill (>1) vs decode (==1). Without it the
        // spine-side wire timings could not be split by phase either, so joining
        // against the worker log to get `wire = ns_blocked - worker_service` gave
        // one blended number across two workloads that differ by ~500x in tokens.
        //
        // ns_before_await is issue -> the moment we START awaiting this request;
        // the spine is doing its own work (issuing others, awaiting an earlier
        // worker) during it. ns_blocked is the recv itself. THE SPLIT IS THE
        // POINT: per-worker `wait` in the existing stats is issue -> consumed,
        // so for a worker awaited second or third it silently includes time the
        // spine spent on the first, which is why those waits sum to 287 ms
        // against a 156 ms dispatch wall. Only ns_blocked is time genuinely
        // spent waiting on the wire and the worker.
        //
        // Join offline against the worker's own WP_REQ_LOG ns_wall (same request
        // order per worker) to get wire = ns_blocked - worker_service.
        const auto wp_await_t0 = req_log_ != nullptr ? dispatch_clock::now()
                                                     : dispatch_clock::time_point();
        request.await_started_at = wp_await_t0;
        const dispatch_clock::time_point recv_started =
            layer_trace_enabled() ? dispatch_clock::now() : dispatch_clock::time_point{};
        uint64_t         wanted_seq_id = seq_id;
        pipe_frame_type  type          = await_response(request, wanted_seq_id, payload, state);
        const bool       streamed      = type == PIPE_EXPERT_PARTIAL_STREAM;
        if (!streamed) {
            note_in_flight_delta(state, -1);
        }
        dispatch_clock::time_point response_received_at = dispatch_clock::now();
        // WP_DISPATCH_DEDUP_ACTIVATIONS FALLBACK. A secondary sent
        // PIPE_EXPERT_DISPATCH_ACTS_REF answers PIPE_ERR_ACTS_UNAVAILABLE,
        // never a partial, when its bounded local wait on the shm segment came
        // up empty (see the frame's handler in wp-expert-worker.cpp). That is
        // the ONE failure mode this mechanism is allowed to have: the primary
        // already confirmed (PIPE_EXPERT_ACTS_PUBLISH_ACK, awaited BEFORE this
        // REF was ever sent -- see dedup_publish_and_ref()) that the segment
        // was published and ready, so a secondary that still cannot read it is
        // hitting a purely LOCAL fault (permissions, memory pressure, a kernel
        // shm limit) with no route to a correct answer over this frame. Retry
        // ONCE, same worker, ordinary inline bytes -- exactly what would have
        // been sent had dedup never been attempted for this one request. This
        // is the ONLY place fallback happens; it is deliberately NOT triggered
        // by a general PIPE_ERROR (a real protocol/compute error on a normal
        // request must still throw, never silently retry with different
        // bytes) and it is bounded to one attempt so a wedged worker still
        // fails loudly rather than looping.
        if (type == PIPE_ERROR && request.dedup_role_secondary && !request.dedup_retried) {
            const pipe_error probe = pipe_decode_error(payload.data(), payload.size());
            if (probe.code == (uint32_t) PIPE_ERR_ACTS_UNAVAILABLE) {
                request.dedup_retried = true;
                worker & value = workers[request.worker_index];
                // Bit-exactness: dedup_fallback_acts_payload was encoded from the
                // SAME `activations` slice the primary published, at plan time --
                // this is not a re-derivation, it is the untouched fallback copy
                // this mechanism exists to make available. See
                // dedup_publish_and_ref() for where it is populated.
                const uint64_t retry_seq = wanted_seq_id | (1ull << 63);
                note_in_flight_delta(state, +1);
                const bool sent =
                    pipe_send_frame(*value.socket, PIPE_EXPERT_DISPATCH_BEGIN, retry_seq,
                                    request.begin_payload.data(), request.begin_payload.size()) &&
                    pipe_send_frame(*value.socket, PIPE_EXPERT_DISPATCH_ACTS, retry_seq,
                                    request.dedup_fallback_acts_payload.data(),
                                    request.dedup_fallback_acts_payload.size());
                if (!sent) {
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " failed to send the dedup fallback for expert(s) " +
                                             assignment_experts(request.assignments));
                }
                wanted_seq_id = retry_seq;
                type = await_response(request, wanted_seq_id, payload, state);
                if (type != PIPE_EXPERT_PARTIAL_STREAM) {
                    note_in_flight_delta(state, -1);
                }
                response_received_at = dispatch_clock::now();
            }
        }
        request.await_finished_at = req_log_ != nullptr ? response_received_at
                                                         : dispatch_clock::time_point();
        request.response_bytes = req_log_ != nullptr && type != PIPE_EXPERT_PARTIAL_STREAM
            ? payload.size() : 0;
        if (collect_stats && last_response != nullptr) {
            *last_response = response_received_at;
            // per-request wait is only tracked for the primary wait loop via stats.workers
        }
        if (speed_split) {
            request.wait_ns = elapsed_ns(request.issued_at, dispatch_clock::now());
        }
        worker & value = workers[request.worker_index];
        if (type == PIPE_ERROR) {
            const pipe_error error = pipe_decode_error(payload.data(), payload.size());
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                     " rejected expert(s) " + assignment_experts(request.assignments) +
                                     " on layer " + std::to_string(layer) + " with code " +
                                     std::to_string(error.code) + ": " + error.msg);
        }
        if (type != PIPE_EXPERT_PARTIAL && type != PIPE_EXPERT_PARTIAL_STREAM) {
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                     " returned frame type " + std::to_string((uint32_t) type) +
                                     " for expert(s) " + assignment_experts(request.assignments));
        }

        if (type == PIPE_EXPERT_PARTIAL_STREAM) {
            std::vector<std::vector<float>> partials;
            std::vector<uint8_t> received;
            uint32_t part_count = 0;
            size_t n_received = 0;
            size_t next_fold = 0;
            dispatch_clock::time_point first_fold_at;
            dispatch_clock::time_point last_ready_at = response_received_at;
            const uint32_t want_rows = request.token_ids.empty()
                ? n_tokens : (uint32_t) request.token_ids.size();
            const size_t want_vals = (size_t) want_rows * (size_t) n_embd;
            out.assign(want_vals, 0.0f);
            for (;;) {
                const dispatch_clock::time_point frame_ready_at = response_received_at;
                if (type != PIPE_EXPERT_PARTIAL_STREAM) {
                    note_in_flight_delta(state, -1);
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " interleaved a non-stream frame while sending partials");
                }
                pipe_expert_partial_stream stream_partial;
                try {
                    const dispatch_clock::time_point decode_started =
                        layer_trace_enabled() ? dispatch_clock::now() : dispatch_clock::time_point{};
                    stream_partial = pipe_decode_expert_partial_stream(
                        payload.data(), payload.size(), n_embd);
                    if (layer_trace_enabled()) {
                        add_layer_trace(layer, &layer_trace_stats::decode_ns,
                                        elapsed_ns(decode_started, dispatch_clock::now()));
                    }
                } catch (const std::exception & error) {
                    note_in_flight_delta(state, -1);
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " returned an invalid streamed partial for expert(s) " +
                                             assignment_experts(request.assignments) + ": " + error.what());
                }
                if (stream_partial.part_count > 64) {
                    note_in_flight_delta(state, -1);
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " returned too many streamed partials");
                }
                if (part_count == 0) {
                    part_count = stream_partial.part_count;
                    partials.resize(part_count);
                    received.assign(part_count, 0);
                } else if (stream_partial.part_count != part_count) {
                    note_in_flight_delta(state, -1);
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " changed streamed partial count");
                }
                const size_t part_index = stream_partial.part_index;
                if (received[part_index] != 0) {
                    note_in_flight_delta(state, -1);
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " repeated a streamed partial");
                }
                const pipe_expert_partial & partial = stream_partial.partial;
                if (partial.layer != layer || partial.n_tokens != want_rows ||
                    partial.partial.size() != want_vals) {
                    note_in_flight_delta(state, -1);
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " returned the wrong streamed partial shape for expert(s) " +
                                             assignment_experts(request.assignments));
                }
                for (size_t i = 0; i < partial.partial.size(); ++i) {
                    if (std::isfinite(partial.partial[i])) {
                        continue;
                    }
                    note_in_flight_delta(state, -1);
                    const size_t row = i / (size_t) n_embd;
                    const uint32_t token = request.token_ids.empty()
                        ? (uint32_t) row : request.token_ids[row];
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " returned a NON-FINITE streamed partial at layer " +
                                             std::to_string(layer) + " row " + std::to_string(row) +
                                             " (token " + std::to_string(token) + ") dim " +
                                             std::to_string(i % (size_t) n_embd) + " for expert(s) " +
                                             assignment_experts(request.assignments));
                }
                partials[part_index] = std::move(stream_partial.partial.partial);
                received[part_index] = 1;
                ++n_received;
                request.response_bytes += payload.size();
                last_ready_at = frame_ready_at;
                while (next_fold < part_count && received[next_fold] != 0) {
                    const bool early = n_received < part_count;
                    if (collect_stats && first_fold_at == dispatch_clock::time_point()) {
                        first_fold_at = dispatch_clock::now();
                    }
                    scatter_add(out, partials[next_fold], request);
                    if (collect_stats && early) {
                        ++state.stats.n_partials_folded_early;
                    }
                    ++next_fold;
                }
                if (n_received == part_count) {
                    note_in_flight_delta(state, -1);
                    if (collect_stats && first_fold_at != dispatch_clock::time_point() &&
                        last_ready_at > first_fold_at) {
                        state.stats.ns_fold_overlapped += elapsed_ns(first_fold_at, last_ready_at);
                    }
                    break;
                }
                type = await_response(request, wanted_seq_id, payload, state);
                response_received_at = dispatch_clock::now();
            }
            request.await_finished_at = req_log_ != nullptr ? response_received_at
                                                             : dispatch_clock::time_point();
            if (collect_stats && last_response != nullptr) {
                *last_response = response_received_at;
            }
            if (layer_trace_enabled()) {
                add_layer_trace(layer, &layer_trace_stats::recv_ns,
                                elapsed_ns(recv_started, dispatch_clock::now()));
            }
            if (dispatch_hash_trace_enabled()) {
                const uint64_t hash = dispatch_hash_fnv1a(out.data(), out.size() * sizeof(float));
                if (value.inproc) {
                    std::fprintf(stderr,
                                 "DISPPART seq=%llu layer=%d worker=inproc h=%llu\n",
                    (unsigned long long) state.dispatch_hash_seq_, layer,
                                 (unsigned long long) hash);
                } else {
                    std::fprintf(stderr,
                                 "DISPPART seq=%llu layer=%d worker=%zu h=%llu\n",
                                 (unsigned long long) state.dispatch_hash_seq_, layer,
                                 request.worker_index, (unsigned long long) hash);
                }
            }
            GGML_ASSERT(out.size() == want_vals);
            GGML_UNUSED(n_values);
            return;
        }

        if (layer_trace_enabled()) {
            add_layer_trace(layer, &layer_trace_stats::recv_ns,
                            elapsed_ns(recv_started, dispatch_clock::now()));
        }
        pipe_expert_partial partial;
        try {
            const dispatch_clock::time_point decode_started =
                layer_trace_enabled() ? dispatch_clock::now() : dispatch_clock::time_point{};
            partial = pipe_decode_expert_partial(payload.data(), payload.size(), n_embd);
            if (layer_trace_enabled()) {
                add_layer_trace(layer, &layer_trace_stats::decode_ns,
                                elapsed_ns(decode_started, dispatch_clock::now()));
            }
        } catch (const std::exception & error) {
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                     " returned an invalid partial for expert(s) " +
                                     assignment_experts(request.assignments) + ": " + error.what());
        }
        // Partial carries (layer, n_tokens); token identity is the layout of
        // partial[token * n_embd + dim]. Do not rely on arrival ordering across
        // workers — each partial is a full [n_tokens * n_embd] block.
        //
        // Under the spine-side gather this request may have carried only a subset
        // of rows, so the shape we expect back is THIS REQUEST'S row count, not
        // the layer's. token_ids empty = identity = the layer's n_tokens.
        const uint32_t want_rows = request.token_ids.empty()
            ? n_tokens : (uint32_t) request.token_ids.size();
        const size_t   want_vals = (size_t) want_rows * (size_t) n_embd;
        if (partial.layer != layer || partial.n_tokens != want_rows || partial.partial.size() != want_vals) {
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                     " returned the wrong partial shape for expert(s) " +
                                     assignment_experts(request.assignments) +
                                     " (layer=" + std::to_string(partial.layer) +
                                     " want=" + std::to_string(layer) +
                                     " n_tokens=" + std::to_string(partial.n_tokens) +
                                     " want_n_tokens=" + std::to_string(want_rows) + ")");
        }
        if (dispatch_hash_trace_enabled()) {
            const uint64_t hash = dispatch_hash_fnv1a(
                partial.partial.data(), partial.partial.size() * sizeof(float));
            if (value.inproc) {
                std::fprintf(stderr,
                             "DISPPART seq=%llu layer=%d worker=inproc h=%llu\n",
                             (unsigned long long) state.dispatch_hash_seq_, layer,
                             (unsigned long long) hash);
            } else {
                std::fprintf(stderr,
                             "DISPPART seq=%llu layer=%d worker=%zu h=%llu\n",
                             (unsigned long long) state.dispatch_hash_seq_, layer,
                             request.worker_index, (unsigned long long) hash);
            }
        }
        // `partial.partial` is ALWAYS f32 here regardless of what dtype the worker
        // put on the wire (PIPE_VERSION 13's self-describing dtype tag): the tag
        // lives on the frame and pipe_decode_expert_partial() does the fp16->fp32
        // widening internally before this function ever sees the vector. The spine
        // does not need to know or configure anything about a worker's dtype choice
        // -- it only ever operates on f32, and scatter_add below sums in f32 either
        // way. Current workers only ever send f32 (WP_EXPERT_PARTIAL_DTYPE=f16 was
        // removed 2026-08-19, see pipe-protocol.h), but the decode path still
        // accepts f16 from a stale worker mid-rolling-restart, so this comment and
        // the code below make no assumption about which one arrives.
        // MAD-LAB DIAGNOSTIC: the spine validates every weight it SENDS (see
        // pipe-protocol.cpp, "expert dispatch has a non-finite weight") but never
        // validated a partial it RECEIVES. That asymmetry let a worker return
        // NaN rows that landed silently in ffn_moe_out via scatter_add, and only
        // surfaced one layer later as a bogus "non-finite weight" rejection --
        // blaming the spine's routing for the previous layer's corrupted output.
        // Name the worker and the row instead.
        for (size_t i = 0; i < partial.partial.size(); ++i) {
            if (std::isfinite(partial.partial[i])) {
                continue;
            }
            const size_t row = i / (size_t) n_embd;
            const uint32_t token = request.token_ids.empty()
                ? (uint32_t) row : request.token_ids[row];
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                     " returned a NON-FINITE partial at layer " +
                                     std::to_string(layer) + " row " + std::to_string(row) +
                                     " (token " + std::to_string(token) + ") dim " +
                                     std::to_string(i % (size_t) n_embd) +
                                     " for expert(s) " + assignment_experts(request.assignments));
        }
        out.assign(partial.partial.begin(), partial.partial.end());
        GGML_ASSERT(out.size() == want_vals);
        GGML_UNUSED(n_values);
    }

    // Original behaviour, kept for the deferred-fold path: receive and add.
    void accumulate_partial(std::vector<float> &             result,
                            planned_request &                request,
                            uint64_t                         seq_id,
                            int32_t                          layer,
                            uint32_t                         n_tokens,
                            dispatch_clock::time_point *     last_response,
                            dispatch_state &                 state) {
        std::vector<float> one;
        receive_partial(one, result.size(), request, seq_id, layer, n_tokens, last_response, state);
        const auto unpack_t0 = req_log_ != nullptr ? dispatch_clock::now()
                                                   : dispatch_clock::time_point();
        scatter_add(result, one, request);
        if (req_log_ != nullptr) {
            request.unpack_ns = elapsed_ns(unpack_t0, dispatch_clock::now());
            write_request_log(request, layer, n_tokens, state);
        }
    }

    // Add a worker's partial into the layer result. Identity (token_ids empty) is
    // a straight elementwise add; under the gather the partial's row r belongs to
    // original token token_ids[r]. Rows no worker asked for are left untouched,
    // which is correct because they had a zero routing weight everywhere and the
    // caller zero-initialises `result`.
    void scatter_add(std::vector<float> &      result,
                     const std::vector<float> & one,
                     const planned_request &    request) const {
        // WP_SIMD_UNPACK: vectorized f32 accumulate (bit-identical to the scalar
        // add; only the per-element add is vectorized, cross-partial sum ORDER is
        // still set by the caller's fixed request order, unchanged). Default off.
        const bool simd = pipe_simd_unpack_enabled() != 0;
        if (request.token_ids.empty()) {
            GGML_ASSERT(one.size() == result.size());
            if (simd) {
                pipe_simd_accumulate_f32(result.data(), one.data(), result.size());
            } else {
                for (size_t i = 0; i < result.size(); ++i) {
                    result[i] += one[i];
                }
            }
            return;
        }
        const size_t width = (size_t) n_embd;
        for (size_t r = 0; r < request.token_ids.size(); ++r) {
            const size_t dst = (size_t) request.token_ids[r] * width;
            const size_t src = r * width;
            GGML_ASSERT(dst + width <= result.size());
            if (simd) {
                pipe_simd_accumulate_f32(result.data() + dst, one.data() + src, width);
            } else {
                for (size_t d = 0; d < width; ++d) {
                    result[dst + d] += one[src + d];
                }
            }
        }
    }

    // Poll ALL outstanding requests' sockets and receive each partial the
    // moment it arrives, returning partials[i] indexed by REQUEST i (fixed
    // order), not arrival order. Shared by harvest_partials (this layer's
    // immediate requests) and collect_pending_deferred (the previous layer's
    // deferred requests) -- both need the identical arrival-order-receive,
    // fixed-order-reduce shape; see the WHY / WHY SUM IN FIXED ORDER notes on
    // harvest_partials below, which apply here unchanged.
    std::vector<std::vector<float>> poll_harvest_receive(std::vector<planned_request> & requests,
                                                          uint64_t                       seq_id,
                                                          int32_t                        layer,
                                                          uint32_t                       n_tokens,
                                                          dispatch_clock::time_point *   last_response,
                                                          dispatch_state &               state) {
        const size_t n = requests.size();
        std::vector<std::vector<float>> partials(n);
        if (n == 0) {
            return partials;
        }
        std::vector<char>               done(n, 0);
        size_t                          remaining = n;

        while (remaining > 0) {
            // Poll set = the FIRST outstanding request per socket. Two requests
            // sharing a worker must stay in FIFO order on that socket, and
            // await_response's seq_id check would throw if they were reordered.
            std::vector<struct pollfd> pfds;
            std::vector<size_t>        idx;
            std::set<int>              seen;
            for (size_t i = 0; i < n; ++i) {
                if (done[i]) {
                    continue;
                }
                worker & w = workers[requests[i].worker_index];
                if (!w.socket) {
                    pfds.clear();
                    idx.clear();
                    break;
                }
                const int fd = w.socket->poll_fd();
                if (fd < 0) {
                    pfds.clear();
                    idx.clear();
                    break;
                }
                if (!seen.insert(fd).second) {
                    continue;
                }
                struct pollfd p;
                p.fd      = fd;
                p.events  = POLLIN;
                p.revents = 0;
                pfds.push_back(p);
                idx.push_back(i);
            }
            if (pfds.empty()) {
                // No pollable descriptor: fall back to the original fixed-order
                // await so this can never be worse than what it replaced.
                for (size_t i = 0; i < n; ++i) {
                    if (done[i]) {
                        continue;
                    }
                    receive_partial(partials[i], 0, requests[i], seq_id,
                                    layer, n_tokens, last_response, state);
                    done[i] = 1;
                    --remaining;
                }
                break;
            }
            const int r = ::poll(pfds.data(), (nfds_t) pfds.size(), -1);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(std::string("expert dispatcher poll failed: ") +
                                         std::strerror(errno));
            }
            for (size_t k = 0; k < pfds.size(); ++k) {
                if (pfds[k].revents == 0) {
                    continue;
                }
                const size_t i = idx[k];
                receive_partial(partials[i], 0, requests[i], seq_id,
                                layer, n_tokens, last_response, state);
                done[i] = 1;
                --remaining;
            }
        }

        return partials;
    }

    // Harvest a layer's partials AS THEY ARRIVE rather than in fixed worker
    // order, then sum them in fixed order.
    //
    // WHY. Measured 2026-08-02: 149.19 of the 155.8 ms/token dispatch wall is
    // spent genuinely blocked, but the spine awaited worker 0, then 1, then 2,
    // so a worker that had already answered sat unread until its turn came. The
    // per-request log showed worker 1 blocking 9.6 us -- its response had been
    // sitting in the socket the whole time. Per layer that cost ~1.6 ms beyond
    // the slowest worker's own service, ~69 ms/token.
    //
    // WHY SUM IN FIXED ORDER. Floating-point addition is not associative, so
    // summing in arrival order would make the result depend on network timing.
    // Buffering costs 3 x n_embd floats and removes a source of run-to-run
    // variance rather than adding one. (Worker ASSIGNMENT is already timing-
    // dependent -- ~35% of requests differ between identical runs -- but that is
    // no reason to add a second such source here.)
    void harvest_partials(std::vector<float> &             result,
                          std::vector<planned_request> &   requests,
                          uint64_t                         seq_id,
                          int32_t                          layer,
                          uint32_t                         n_tokens,
                          dispatch_clock::time_point *     last_response,
                          dispatch_state &                 state) {
        const size_t n = requests.size();
        if (n == 0) {
            return;
        }
        std::vector<std::vector<float>> partials =
                poll_harvest_receive(requests, seq_id, layer, n_tokens, last_response, state);

        // Fixed request order, not arrival order -- see the note above on why the
        // sum must not depend on network timing. scatter_add keeps that property:
        // a row touched by several workers is still summed in request order.
        for (size_t i = 0; i < n; ++i) {
            const auto unpack_t0 = req_log_ != nullptr ? dispatch_clock::now()
                                                       : dispatch_clock::time_point();
            scatter_add(result, partials[i], requests[i]);
            if (req_log_ != nullptr) {
                requests[i].unpack_ns = elapsed_ns(unpack_t0, dispatch_clock::now());
                write_request_log(requests[i], layer, n_tokens, state);
            }
        }
    }

    // Collect previously-issued deferred partials and sum them. Caller must
    // already have issued the current layer's requests so N-1 deferred overlaps
    // N's in-flight reads. Marks late if the fold point was already closed.
    std::vector<float> collect_pending_deferred(bool mark_fold_open, dispatch_state & state) {
        if (pending_def.requests.empty()) {
            return {};
        }
        if (mark_fold_open) {
            pending_def.fold_opened = true;
        }
        const size_t           n_values = (size_t) pending_def.n_tokens * (size_t) n_embd;
        std::vector<float>     fold(n_values, 0.0f);
        const int32_t          layer  = pending_def.layer;
        const uint64_t         seq_id = pending_def.seq_id;
        const uint32_t         n_tok  = pending_def.n_tokens;
        std::vector<planned_request> requests = std::move(pending_def.requests);
        pending_def.requests.clear();

        // WP_DISPATCH_HARVEST=1 or WP_UNPACK_OVERLAP=1: the deferred fold has
        // the identical serial-await shape harvest_partials fixes for
        // immediate requests -- N-1's deferred requests were also issued to
        // every slice worker, then awaited worker-by-worker here. Route
        // through the same poll-and-reduce-in-fixed-order helper so both
        // await paths agree under either flag; default (both off) keeps this
        // loop's prior behaviour untouched.
        if (harvest_enabled() || unpack_overlap) {
            std::vector<std::vector<float>> partials =
                poll_harvest_receive(requests, seq_id, layer, n_tok, nullptr, state);
            for (size_t i = 0; i < requests.size(); ++i) {
                planned_request & request = requests[i];
                // If the successor layer already returned without this partial, it is late.
                if (pending_def.fold_closed) {
                    ++deferral.n_deferred_late;
                }
                scatter_add(fold, partials[i], request);
                update_speed_estimate(request);
                update_residency(request.worker_index, layer, request.assignments);
            }
        } else {
            for (planned_request & request : requests) {
                // If the successor layer already returned without this partial, it is late.
                if (pending_def.fold_closed) {
                    ++deferral.n_deferred_late;
                }
                accumulate_partial(fold, request, seq_id, layer, n_tok, nullptr, state);
                update_speed_estimate(request);
                update_residency(request.worker_index, layer, request.assignments);
            }
        }
        pending_def = {};
        return fold;
    }

    static constexpr size_t k_max_open_dispatches = 2;

    // Guards dispatch_slots_ occupancy (allocate / look up / release) and
    // next_dispatch_handle_ only -- NOT the blocking wait in finish_dispatch()
    // (await_response() and friends), which runs after a handle's requests
    // are already on the wire. The lock is held only for slot metadata; the
    // split exists so a second handle can begin (mutate the map)
    // while the first handle's finish is parked in a blocking recv, instead
    // of one mutex forcing that recv to also block out a concurrent begin.
    std::mutex                                        dispatch_map_mutex_;
    std::array<dispatch_state, k_max_open_dispatches> dispatch_slots_{};
    dispatch_handle                                   next_dispatch_handle_ = 1;

    // Peek at slot availability WITHOUT claiming one. Throws the same
    // "already open" error when every slot is taken. Called at the very top
    // of begin_dispatch,
    // mirroring the old check's position before any of begin_dispatch's own
    // validation (route lookup, activation shape, assignment sanity) or its
    // try block: this must not mutate dispatch_slots_, because those
    // validation throws are OUTSIDE the try/catch(poison) below and must
    // leave the dispatcher exactly as reusable as they always did -- if this
    // peek instead claimed a slot up front, one of those unrelated throws
    // would leak a permanently "open" slot that nothing ever releases.
    void check_dispatch_capacity() {
        std::lock_guard<std::mutex> lock(dispatch_map_mutex_);
        for (const dispatch_state & slot : dispatch_slots_) {
            if (slot.handle == k_invalid_handle) {
                return;
            }
        }
        throw std::runtime_error("expert dispatcher begin_dispatch called while a dispatch is already open");
    }

    // Actually claim a free slot. Called only once begin_dispatch's own
    // validation and planning have fully succeeded, at the same point the
    // old singleton was set open (right before returning) -- so a handle
    // only exists once a dispatch is truly in flight, exactly as `open_disp
    // .open` only ever became true there. Unreachable in practice (capacity
    // was already confirmed by check_dispatch_capacity(), and calls on one
    // dispatcher are serialized), but throws rather than asserts so a future
    // concurrent caller fails loudly instead of corrupting a slot.
    dispatch_handle acquire_dispatch_slot() {
        std::lock_guard<std::mutex> lock(dispatch_map_mutex_);
        for (dispatch_state & slot : dispatch_slots_) {
            if (slot.handle == k_invalid_handle) {
                slot.handle = next_dispatch_handle_++;
                return slot.handle;
            }
        }
        throw std::runtime_error("expert dispatcher begin_dispatch called while a dispatch is already open");
    }

    // dispatch_slots_ is fixed-size and tiny (k_max_open_dispatches stays a
    // handful even at stage 4), so a linear scan is simpler than a real map
    // and just as cheap at this size.
    dispatch_state * find_dispatch_slot(dispatch_handle handle) {
        std::lock_guard<std::mutex> lock(dispatch_map_mutex_);
        for (dispatch_state & slot : dispatch_slots_) {
            if (slot.handle == handle) {
                return &slot;
            }
        }
        return nullptr;
    }

    dispatch_stats stats_for(dispatch_handle handle) {
        std::lock_guard<std::mutex> lock(dispatch_map_mutex_);
        for (const dispatch_state & slot : dispatch_slots_) {
            if (slot.handle == handle) {
                return slot.stats;
            }
        }
        return {};
    }

    void release_dispatch_slot(dispatch_handle handle) {
        std::lock_guard<std::mutex> lock(dispatch_map_mutex_);
        for (dispatch_state & slot : dispatch_slots_) {
            if (slot.handle == handle) {
                slot = dispatch_state{};
                return;
            }
        }
    }

    // Legacy no-arg finish_dispatch()/has_open_dispatch() need to know WHICH
    // handle is open without being told. The graph uses explicit handles when
    // two dispatches are live.
    dispatch_handle only_open_handle() {
        std::lock_guard<std::mutex> lock(dispatch_map_mutex_);
        for (const dispatch_state & slot : dispatch_slots_) {
            if (slot.handle != k_invalid_handle) {
                return slot.handle;
            }
        }
        return k_invalid_handle;
    }

    bool has_open_dispatch_slot() {
        return only_open_handle() != k_invalid_handle;
    }

    // Demultiplex a received frame's seq_id to the handle it belongs to. The
    // dedup-retry path (see receive_partial's retry_seq) resends with the
    // same seq_id OR'd with the high bit, so mask that off before matching a
    // slot's registered seq_id -- otherwise a retried response would demux to
    // "unknown" even though it answers the same handle's request. Returns
    // k_invalid_handle when no open slot claims it. The strict sequence check
    // in await_response remains the final ordering check for each handle.
    dispatch_handle demux_seq_id(uint64_t received_seq_id) {
        const uint64_t masked = received_seq_id & ~(uint64_t(1) << 63);
        std::lock_guard<std::mutex> lock(dispatch_map_mutex_);
        for (const dispatch_state & slot : dispatch_slots_) {
            if (slot.handle != k_invalid_handle && slot.seq_id == masked) {
                return slot.handle;
            }
        }
        return k_invalid_handle;
    }

    dispatch_handle begin_dispatch(int32_t                          layer,
                        uint64_t                                    seq_id,
                        uint32_t                                    n_tokens,
                        const std::vector<float> &                  activations,
                        const std::vector<pipe_expert_assignment> & assignments,
                        float                                       swiglu_clamp,
                        uint32_t                                    chunk_index,
                        const std::vector<pipe_expert_assignment> * layer_assignments,
                        uint32_t                                    layer_n_tokens) {
        if (!has_open_dispatch_slot()) {
            reset_layer_trace(layer);
        }
        if (poisoned) {
            throw std::runtime_error("expert dispatcher cannot be reused after a worker or protocol failure");
        }
        check_dispatch_capacity();
        if (chunk_index >= (uint32_t) dispatch_chunks_) {
            throw std::invalid_argument("expert dispatcher has an invalid chunk index");
        }
        const auto route_it = routes.find(layer);
        if (route_it == routes.end()) {
            throw std::invalid_argument("expert dispatcher has no workers for layer " + std::to_string(layer));
        }
        const uint64_t activation_count = (uint64_t) n_tokens * (uint64_t) n_embd;
        if (n_tokens == 0 || activation_count != activations.size()) {
            throw std::invalid_argument("expert dispatcher activation shape does not match n_tokens and n_embd");
        }
        if (assignments.empty()) {
            throw std::invalid_argument("expert dispatcher requires at least one activated expert");
        }

        const bool hash_trace = dispatch_hash_trace_enabled();
        const uint64_t hash_seq = hash_trace
            ? g_dispatch_hash_seq.fetch_add(1, std::memory_order_relaxed) : 0;
        const uint64_t hash_pre = hash_trace
            ? dispatch_hash_fnv1a(activations.data(), activations.size() * sizeof(float)) : 0;

        std::set<int32_t> seen_experts;
        for (const pipe_expert_assignment & assignment : assignments) {
            if (assignment.expert_id < 0 || assignment.expert_id >= n_expert ||
                !seen_experts.insert(assignment.expert_id).second || assignment.weights.size() != n_tokens) {
                throw std::invalid_argument("expert dispatcher has an invalid or repeated expert assignment");
            }
        }
        if (chunk_index == 0) {
            const std::vector<pipe_expert_assignment> & routed =
                layer_assignments != nullptr ? *layer_assignments : assignments;
            const uint32_t routed_tokens = layer_n_tokens != 0 ? layer_n_tokens : n_tokens;
            add_temporal_locality(layer, routed_tokens, routed);
            dump_routing(layer, routed_tokens, routed);
        }

        try {
            const dispatch_handle handle = acquire_dispatch_slot();
            dispatch_state & state = *find_dispatch_slot(handle);
            state.layer              = layer;
            state.chunk_index        = chunk_index;
            state.n_tokens           = n_tokens;
            state.seq_id             = seq_id;
            state.activation_count   = activation_count;
            state.dispatch_hash_seq_ = hash_seq;
            state.hash_pre           = hash_pre;
            state.req_dispatch_start_ = req_log_ != nullptr ? dispatch_clock::now()
                                                             : dispatch_clock::time_point{};
            // Decide whether this layer may leave experts deferred.
            // The last main-graph MoE layer has no successor to fold into — do
            // not defer it. Prefer the host-provided last_no_defer_layer
            // (hparams.n_layer()-1) over the worker HELLO max: workers also
            // advertise NextN/MTP layers (e.g. 78) that the main graph never
            // dispatches, which previously left every token's last main MoE
            // layer deferred and drained as n_deferred_late at end_decode.
            const int32_t no_defer_layer =
                last_no_defer_layer >= 0 ? last_no_defer_layer : last_routed_layer;
            // Decode/spec-verify only, gated on WIDTH (n_tokens <=
            // defer_max_width_, default 32 via WP_DEFER_MAX_WIDTH) rather than
            // n_tokens == 1. Prefill dispatches the full ubatch (2048 tokens);
            // decode is a spec-decode VERIFY BATCH of 1 + up to 7 draft tokens
            // -- never a single token -- so n_tokens == 1 was never true here
            // and previously left WP_DEFER_K permanently inert. See
            // defer_max_width_enabled() above for the width-vs-exact-one
            // rationale; a batch wider than one token is not thereby a
            // prompt, and this conflation has now been found three times in
            // this codebase. Prefill shrinks the last layer via
            // get_rows(out_ids) so a deferred partial from layer L with the
            // full prefill width cannot fold into layer L+1's MoE output —
            // that was a silent drop + n_deferred_late path. Spec allows
            // disabling prefill rather than half-working it; the width gate
            // is what keeps prefill (2048 tokens) out of deferral, same as
            // n_tokens == 1 did, without also excluding every decode step.
            const bool may_defer =
                defer_k_value > 0 &&
                layer != no_defer_layer &&
                n_tokens <= defer_max_width_;

            std::vector<pipe_expert_assignment> immediate;
            std::vector<pipe_expert_assignment> deferred;
            size_t                              n_def = 0;
            if (may_defer) {
                split_immediate_deferred(assignments, n_tokens, defer_k_value, immediate, deferred, n_def);
            } else {
                immediate = assignments;
            }
            deferral.n_deferred += n_def;

            const dispatch_clock::time_point issue_start =
                collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};

            // Shared assigned_counts so residency balancing sees both sets.
            // Note: plan runs BEFORE collect_pending_deferred, so choose_worker
            // does not yet see residency updates from the previous layer's
            // deferred drain. Residency affinity is a heuristic, not a
            // correctness property; this only shifts when deferred experts
            // refresh the LRU relative to the prior order.
            std::vector<size_t> assigned_counts(workers.size(), 0);
            std::vector<planned_request> imm_requests =
                plan_requests(layer, n_tokens, activations, immediate, route_it->second, assigned_counts,
                              swiglu_clamp);
            std::vector<planned_request> def_requests =
                deferred.empty()
                    ? std::vector<planned_request>{}
                    : plan_requests(layer, n_tokens, activations, deferred, route_it->second, assigned_counts,
                                    swiglu_clamp);

            state.stats              = {};
            state.stats.workers_used = imm_requests.size() + def_requests.size();
            for (const planned_request & request : imm_requests) {
                state.stats.workers.push_back({
                    workers[request.worker_index].info.endpoint,
                    request.assignments.size(),
                    0,
                    1,
                    request.assignments.size(),
                });
            }

            // Occupancy ordering (fetch-against-fetch):
            //   1. issue layer N immediate AND deferred   <- first
            //   2. await layer N-1 deferred               <- overlaps with (1)
            //   3. await layer N immediate
            //   4. fold N-1 deferred into result, return
            // Issuing N before collecting N-1 is the whole occupancy win: N-1's
            // deferred reads stay in flight while N's reads are also in flight.
            // Collecting first would serialise them (wait N-1 def, then issue N)
            // and leave nvme_util_pct unchanged.
            //
            // Per-worker send order within a layer: all immediate requests, then
            // all deferred. Both batches share this layer's seq_id; TCP FIFO per
            // socket plus await_response's seq_id check disambiguate. Do not
            // invent a separate seq_id band — a mismatch throws loudly.
            // WP_DISPATCH_DEDUP_ACTIVATIONS: only the immediate set. Deferred
            // requests (WP_DEFER_K > 0) are a decode/spec-verify mechanism
            // (width-gated to n_tokens <= WP_DEFER_MAX_WIDTH, default 32) and
            // dedup is gated to n_tokens > WP_DISPATCH_DEDUP_MIN_TOKENS
            // (default 32) -- the two windows do not overlap in practice, and
            // folding dedup into the deferred-fold's own cross-layer bookkeeping
            // is not a safe thing to do without its own dedicated design.
            dedup_publish_and_ref(imm_requests, seq_id, layer, n_tokens, activations, state);
            issue_requests(imm_requests, seq_id, state);
            if (!def_requests.empty()) {
                issue_requests(def_requests, seq_id, state);
            }
            if (collect_stats) {
                state.stats.ns_issue = elapsed_ns(issue_start, dispatch_clock::now());
            }

            // Drain previous layer's deferred partials now that layer N is in
            // flight. Safe with existing wire format: on any worker socket that
            // carries both, N-1 deferred frames were SENT before N's frames, so
            // they ARRIVE first (TCP FIFO). await_response validates seq_id and
            // throws on mismatch — do not weaken that check.
            std::vector<float> folded_prev = collect_pending_deferred(/*mark_fold_open=*/true, state);

            state.hash_reqs       = imm_requests.size() + def_requests.size();
            state.imm_requests    = std::move(imm_requests);
            state.def_requests    = std::move(def_requests);
            state.folded_prev     = std::move(folded_prev);
            state.assigned_counts = std::move(assigned_counts);
            state.wait_start      = collect_stats ? dispatch_clock::now()
                                                   : dispatch_clock::time_point{};
            last_stats = state.stats;
            return handle;
        } catch (...) {
            poison();
            throw;
        }
    }

    std::vector<float> finish_dispatch() {
        const dispatch_handle handle = only_open_handle();
        if (handle == k_invalid_handle) {
            throw std::runtime_error("expert dispatcher finish_dispatch called with no open dispatch");
        }
        return finish_dispatch(handle, nullptr);
    }

    std::vector<float> finish_dispatch(dispatch_handle handle) {
        return finish_dispatch(handle, nullptr);
    }

    std::vector<float> finish_dispatch(dispatch_handle handle, dispatch_stats * completed_stats) {
        if (poisoned) {
            throw std::runtime_error("expert dispatcher cannot be reused after a worker or protocol failure");
        }
        if (handle == k_invalid_handle) {
            throw std::runtime_error("expert dispatcher finish_dispatch called with no open dispatch");
        }
        dispatch_state * state_ptr       = find_dispatch_slot(handle);
        if (state_ptr == nullptr) {
            throw std::runtime_error("expert dispatcher finish_dispatch called with an invalid handle");
        }
        dispatch_state & slot            = *state_ptr;
        const int32_t  layer            = slot.layer;
        const uint32_t n_tokens         = slot.n_tokens;
        const uint64_t seq_id           = slot.seq_id;
        const uint64_t activation_count = slot.activation_count;
        const uint64_t hash_pre         = slot.hash_pre;
        const size_t   hash_reqs        = slot.hash_reqs;
        std::vector<planned_request> imm_requests    = std::move(slot.imm_requests);
        std::vector<planned_request> def_requests    = std::move(slot.def_requests);
        std::vector<float>           folded_prev     = std::move(slot.folded_prev);
        std::vector<size_t>          assigned_counts = std::move(slot.assigned_counts);
        const dispatch_clock::time_point wait_start  = slot.wait_start;
        // The slot must stay registered through the awaits below: demux_seq_id
        // resolves an incoming response to its owning handle by looking the
        // seq_id up in dispatch_slots_, so releasing before the response has
        // been consumed orphans the seq_id and every await throws "returned
        // sequence N while awaiting N" (gates49, 2026-08-31). Release happens
        // just before the successful return; the failure path is covered by
        // poison() clearing all slots. Stage 4's requirement that a second
        // begin_dispatch not wait on this finish is met by capacity, not by
        // early release.

        try {
            std::vector<float> result((size_t) activation_count, 0.0f);
            // Fold previous deferred into this layer's output (residual path).
            // Partials carry (layer, token) via pending_def.layer + layout; the
            // vectors must agree on n_tokens * n_embd. A mismatch means the
            // fold crossed a token-count boundary (e.g. last-layer get_rows
            // shrink for out_ids during prefill). Never silent-drop: count late
            // and leave a clear trail. Callers that need late==0 must not defer
            // across that boundary (correct last_no_defer_layer + decode n_tokens).
            if (!folded_prev.empty()) {
                if (folded_prev.size() != result.size()) {
                    ++deferral.n_deferred_late;
                    // Consume is already done (partials were awaited in
                    // collect_pending_deferred); we refuse to add a mis-shaped
                    // block into the residual. This is an explicit accounting
                    // path, not a second silent approximation.
                } else {
                    for (size_t i = 0; i < result.size(); ++i) {
                        result[i] += folded_prev[i];
                    }
                }
            }

            dispatch_clock::time_point last_response;
            // See harvest_enabled() for WP_DISPATCH_HARVEST's meaning and the
            // measured default-OFF rationale, and unpack_overlap_enabled()
            // for WP_UNPACK_OVERLAP -- a second, independent opt-in into the
            // SAME poll-and-fixed-order-fold mechanism (poll_harvest_receive/
            // harvest_partials), aimed at overlapping per-worker wire-decode
            // cost with a slower peer's network wait rather than at reducing
            // wait time itself. Either flag takes this branch; both the
            // immediate-request harvest here and the deferred-fold harvest in
            // collect_pending_deferred read the SAME `unpack_overlap` latch
            // (member field, latched once at construction) so a dispatch can
            // never disagree with itself about which await shape is in
            // effect for a given run.
            const bool harvest = harvest_enabled() || unpack_overlap;
            if (harvest) {
                harvest_partials(result, imm_requests, seq_id, layer, n_tokens, &last_response, slot);
                for (size_t request_index = 0; request_index < imm_requests.size(); ++request_index) {
                    planned_request & request = imm_requests[request_index];
                    if (collect_stats) {
                        slot.stats.workers[request_index].ns_wait =
                            elapsed_ns(request.issued_at, last_response);
                    }
                    update_residency(request.worker_index, layer, request.assignments);
                }
            } else {
            for (size_t request_index = 0; request_index < imm_requests.size(); ++request_index) {
                planned_request & request = imm_requests[request_index];
                const dispatch_clock::time_point before = collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};
                accumulate_partial(result, request, seq_id, layer, n_tokens, &last_response, slot);
                if (collect_stats) {
                    slot.stats.workers[request_index].ns_wait = elapsed_ns(request.issued_at, last_response);
                    (void) before;
                }
                update_residency(request.worker_index, layer, request.assignments);
            }
            }
            if (collect_stats && !imm_requests.empty()) {
                slot.stats.ns_wait = elapsed_ns(wait_start, last_response);
            }
            update_speed_estimates(imm_requests);
            log_speed_state(assigned_counts);

            // Stash deferred requests — do NOT wait. They stay in flight until
            // the next layer issues and then collects them (fetch-against-fetch).
            if (!def_requests.empty()) {
                // After the collect above, pending must be empty. Anything still
                // here is a bug — force-collect and mark late before overwriting.
                if (!pending_def.requests.empty()) {
                    pending_def.fold_closed = true;
                    (void) collect_pending_deferred(/*mark_fold_open=*/false, slot);
                }
                pending_def.layer       = layer;
                pending_def.seq_id      = seq_id;
                pending_def.n_tokens    = n_tokens;
                pending_def.requests    = std::move(def_requests);
                pending_def.fold_opened = false;
                pending_def.fold_closed = false;
            }

            if (dispatch_hash_trace_enabled()) {
                const uint64_t hash_post = dispatch_hash_fnv1a(
                    result.data(), result.size() * sizeof(float));
                std::fprintf(stderr,
                             "DISPHASH seq=%llu layer=%d n=%llu pre=%llu post=%llu reqs=%zu\n",
                             (unsigned long long) slot.dispatch_hash_seq_, layer,
                             (unsigned long long) activation_count,
                             (unsigned long long) hash_pre,
                             (unsigned long long) hash_post, hash_reqs);
            }
            if (completed_stats != nullptr) {
                *completed_stats = slot.stats;
            }
            last_stats = slot.stats;
            release_dispatch_slot(handle);
            return result;
        } catch (...) {
            poison();
            throw;
        }
    }

    void begin_window() noexcept {
        window_active     = true;
        window_sample_ok  = sample_nvme_io_ticks(window_io_ticks_begin);
        window_wall_begin = dispatch_clock::now();
        // Reset per-window gap; keep cumulative n_deferred / n_deferred_late.
        deferral.ns_gap        = 0;
        deferral.nvme_util_pct = -1.0;
        gap_at_zero            = (in_flight == 0);
        if (gap_at_zero) {
            gap_zero_since = window_wall_begin;
        }
    }

    void end_window() noexcept {
        if (!window_active) {
            return;
        }
        window_active = false;
        if (gap_at_zero) {
            deferral.ns_gap += elapsed_ns(gap_zero_since, dispatch_clock::now());
            gap_at_zero = false;
        }
        if (window_sample_ok) {
            uint64_t end_ticks = 0;
            if (sample_nvme_io_ticks(end_ticks)) {
                const double wall_ms =
                    std::chrono::duration<double, std::milli>(dispatch_clock::now() - window_wall_begin).count();
                if (wall_ms > 0.0) {
                    const double busy_ms = (double) (end_ticks - window_io_ticks_begin);
                    // Average util across devices is not well-defined when
                    // summing ticks; with one device (the usual case) this is
                    // exact. Cap at 100.
                    double pct = 100.0 * busy_ms / wall_ms;
                    if (pct < 0.0) {
                        pct = 0.0;
                    }
                    if (pct > 100.0) {
                        pct = 100.0;
                    }
                    deferral.nvme_util_pct = pct;
                }
            }
        }
    }

    std::vector<float> drain() {
        if (pending_def.requests.empty()) {
            return {};
        }
        // Anything still pending at drain has missed its fold point.
        pending_def.fold_closed = true;
        dispatch_state state;
        state.layer   = pending_def.layer;
        state.seq_id  = pending_def.seq_id;
        state.n_tokens = pending_def.n_tokens;
        return collect_pending_deferred(/*mark_fold_open=*/false, state);
    }
};

void set_inproc_backend_factory(inproc_backend_factory factory) {
    g_inproc_factory = factory;
}

dispatcher::dispatcher(const std::vector<endpoint> & endpoints) : pimpl(new impl(endpoints)) {}

dispatcher::~dispatcher() = default;

dispatcher::dispatcher(dispatcher &&) noexcept = default;

dispatcher & dispatcher::operator=(dispatcher &&) noexcept = default;

std::vector<float> dispatcher::dispatch(int32_t                                     layer,
                                        uint64_t                                    seq_id,
                                        uint32_t                                    n_tokens,
                                        const std::vector<float> &                  activations,
                                        const std::vector<pipe_expert_assignment> & assignments,
                                        float                                       swiglu_clamp,
                                        uint32_t                                    chunk_index) {
    const dispatch_handle handle =
        pimpl->begin_dispatch(layer, seq_id, n_tokens, activations, assignments, swiglu_clamp, chunk_index,
                              nullptr, 0);
    return pimpl->finish_dispatch(handle);
}

dispatcher::dispatch_handle dispatcher::begin_dispatch(
        int32_t layer, uint64_t seq_id, uint32_t n_tokens,
        const std::vector<float> & activations,
        const std::vector<pipe_expert_assignment> & assignments,
        float swiglu_clamp, uint32_t chunk_index,
        const std::vector<pipe_expert_assignment> * layer_assignments, uint32_t layer_n_tokens) {
    return pimpl->begin_dispatch(layer, seq_id, n_tokens, activations, assignments, swiglu_clamp, chunk_index,
                                 layer_assignments, layer_n_tokens);
}

std::vector<float> dispatcher::finish_dispatch() {
    return pimpl->finish_dispatch();
}

std::vector<float> dispatcher::finish_dispatch(dispatch_handle handle) {
    return pimpl->finish_dispatch(handle);
}

std::vector<float> dispatcher::finish_dispatch(dispatch_handle handle, dispatch_stats * stats) {
    return pimpl->finish_dispatch(handle, stats);
}

bool dispatcher::has_open_dispatch() const {
    return pimpl->has_open_dispatch_slot();
}

bool dispatcher::has_open_dispatch(dispatch_handle handle) const {
    return pimpl->find_dispatch_slot(handle) != nullptr;
}

int dispatcher::dispatch_chunks() const {
    return pimpl->dispatch_chunks_;
}

dispatch_stats dispatcher::stats_for(dispatch_handle handle) const {
    return pimpl->stats_for(handle);
}

layer_trace_stats dispatcher::layer_trace(int32_t layer) const {
    return pimpl->layer_trace(layer);
}

size_t dispatcher::send_prefetch_hints(int32_t layer, const std::vector<int32_t> & experts,
                                       uint32_t provenance, uint32_t n_tokens) {
    return pimpl->send_prefetch_hints(layer, experts, provenance, n_tokens);
}

const prefetch_hint_stats & dispatcher::get_prefetch_hint_stats() const {
    return pimpl->hint_stats;
}

int32_t dispatcher::n_embd() const {
    return pimpl->n_embd;
}

int32_t dispatcher::n_ff_exp() const {
    return pimpl->n_ff_exp;
}

int32_t dispatcher::n_expert() const {
    return pimpl->n_expert;
}

int32_t dispatcher::n_expert_used() const {
    return pimpl->n_expert_used;
}

const std::string & dispatcher::model_identity() const {
    return pimpl->model_identity;
}

const std::vector<worker_info> & dispatcher::workers() const {
    return pimpl->public_workers;
}

size_t dispatcher::in_flight_requests() const {
    return pimpl->in_flight;
}

const dispatch_stats & dispatcher::last_dispatch_stats() const {
    return pimpl->last_stats;
}

const deferral_stats & dispatcher::get_deferral_stats() const {
    return pimpl->deferral;
}

int dispatcher::defer_k() const {
    return pimpl->defer_k_value;
}

int32_t dispatcher::last_no_defer_layer() const {
    return pimpl->last_no_defer_layer >= 0 ? pimpl->last_no_defer_layer : pimpl->last_routed_layer;
}

void dispatcher::set_last_no_defer_layer(int32_t layer) noexcept {
    pimpl->last_no_defer_layer = layer;
}

void dispatcher::begin_deferral_window() noexcept {
    pimpl->begin_window();
}

void dispatcher::end_deferral_window() noexcept {
    pimpl->end_window();
}

std::vector<float> dispatcher::drain_deferred() {
    return pimpl->drain();
}

}  // namespace pipe_expert_dispatcher
