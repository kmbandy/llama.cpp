#include "pipe-expert-dispatch-graph.h"
#include "pipe-prefetch-hints.h"

#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-impl.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

namespace pipe_expert_dispatcher {
namespace {

using dispatch_clock = std::chrono::steady_clock;

bool dispatch_stats_enabled() {
    const char * value = std::getenv("WP_DISPATCH_STATS");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool layer_trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("WP_DS4_LAYER_TRACE");
        return value != nullptr && value[0] != '\0';
    }();
    return enabled;
}

uint64_t elapsed_ns(dispatch_clock::time_point begin, dispatch_clock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

int parse_port(const std::string & text) {
    int        port   = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), port);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() || port <= 0 || port > 65535) {
        throw std::invalid_argument("--expert-dispatch expects host:port[,host:port...]");
    }
    return port;
}

std::vector<endpoint> parse_endpoints(const std::string & text) {
    std::vector<endpoint> result;
    size_t                begin = 0;
    while (begin < text.size()) {
        const size_t      comma = text.find(',', begin);
        const std::string item  = text.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        const size_t      colon = item.rfind(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 == item.size()) {
            throw std::invalid_argument("--expert-dispatch expects host:port[,host:port...]");
        }
        result.push_back({
            item.substr(0, colon),
            parse_port(item.substr(colon + 1)),
            "",
        });
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    if (result.empty() || begin == text.size()) {
        throw std::invalid_argument("--expert-dispatch expects host:port[,host:port...]");
    }
    return result;
}

struct ngram_env_config {
    std::string path;
    int32_t     top_m = 0;
};

ngram_env_config get_ngram_env_config() {
    ngram_env_config result;
    const char * value = std::getenv("WP_HINT_NGRAM");
    if (value == nullptr || value[0] == '\0') {
        return result;
    }
    result.path = value;
    result.top_m = PREFETCH_HINT_MAX_EXPERTS;
    const size_t comma = result.path.rfind(',');
    if (comma == std::string::npos || comma + 1 == result.path.size()) {
        return result;
    }
    int parsed = 0;
    const char * begin = result.path.data() + comma + 1;
    const char * end = result.path.data() + result.path.size();
    const auto number = std::from_chars(begin, end, parsed);
    if (number.ec == std::errc() && number.ptr == end) {
        result.path.resize(comma);
        result.top_m = std::min<int32_t>(PREFETCH_HINT_MAX_EXPERTS, std::max(0, parsed));
    }
    return result;
}

}  // namespace

struct graph_dispatcher::op_context {
    graph_dispatcher * owner = nullptr;
    int32_t            layer = -1;
    // hparams.swiglu_clamp_exp[layer]; <= 0 means no clamp. See the note on
    // pipe_expert_dispatch_req::swiglu_clamp -- the spine's own clamped SwiGLU
    // is unreachable on the dispatch path, so the worker must apply it and the
    // limit has to travel with every request.
    float              swiglu_clamp = 0.0f;
    ggml_tensor *      issued = nullptr;
};

graph_dispatcher::graph_dispatcher(const std::string & endpoints,
                                   int32_t             n_embd,
                                   int32_t             n_ff_exp,
                                   int32_t             n_expert,
                                   int32_t             n_expert_used,
                                   int32_t             last_no_defer_layer) :
    remote(parse_endpoints(endpoints)),
    collect_stats_(dispatch_stats_enabled()) {
    if (layer_trace_enabled()) {
        const char * p = std::getenv("WP_DS4_LAYER_TRACE");
        layer_trace_ = std::fopen(p, "w");
        if (layer_trace_ != nullptr) {
            LLAMA_LOG_WARN("expert dispatch: WP_DS4_LAYER_TRACE=%s (one line per layer)\n", p);
        }
    }
    if (const char * p = std::getenv("WP_FORWARD_LOG"); p != nullptr && p[0] != '\0') {
        forward_log_ = std::fopen(p, "w");
        if (forward_log_ != nullptr) {
            LLAMA_LOG_WARN(
                "expert dispatch: WP_FORWARD_LOG=%s (one line per llama_decode: "
                "n_tokens n_layers ns_wall ns_pack ns_issue ns_wait ns_unpack "
                "ns_dispatch ns_other epoch_end)\n",
                p);
        }
    }
    if (remote.n_embd() != n_embd || remote.n_ff_exp() != n_ff_exp || remote.n_expert() != n_expert ||
        remote.n_expert_used() != n_expert_used) {
        throw std::runtime_error("expert dispatcher workers do not match the model MoE dimensions");
    }

    const ngram_env_config ngram = get_ngram_env_config();
    if (ngram.top_m > 0 && !ngram.path.empty()) {
        try {
            std::unique_ptr<ngram_hint_table> table(new ngram_hint_table(ngram.path));
            if (table->n_experts() != remote.n_expert()) {
                throw std::runtime_error("n-gram table expert count does not match the model");
            }
            if (last_no_defer_layer >= 0 && table->n_layers() > last_no_defer_layer + 1) {
                throw std::runtime_error("n-gram table layer count exceeds the model main stack");
            }
            ngram_top_m_ = std::min<int32_t>(ngram.top_m, table->row_width());
            LLAMA_LOG_WARN("expert dispatch: WP_HINT_NGRAM=%s,%d (%d layers, %zu token-layer rows)\n",
                           ngram.path.c_str(), ngram_top_m_, table->n_layers(), table->row_count());
            ngram_table_ = std::move(table);
        } catch (const std::exception & error) {
            LLAMA_LOG_WARN("expert dispatch: WP_HINT_NGRAM disabled: %s\n", error.what());
        }
    }

    if (last_no_defer_layer >= 0) {
        remote.set_last_no_defer_layer(last_no_defer_layer);
    }

    // Spec §3: print the ACTUAL runtime value at WARN (logger threshold 3 filters
    // INFO; libllama WARN maps to 2 and passes). A gate whose value you cannot
    // see in the log has cost this project three separate retracted measurement sets.
    // Default 0 = feature off = defer nothing. Do NOT add this to struct Options
    // (ABI mismatch 2026-07-30); env var read at startup only.
    const int k = remote.defer_k();
    LLAMA_LOG_WARN(
        "expert dispatch: WP_DEFER_K=%d (%s; 0=off/defer nothing; K=immediate experts per token) "
        "last_no_defer_layer=%d\n",
        k,
        k <= 0 ? "OFF" : "ON",
        remote.last_no_defer_layer());
}

graph_dispatcher::~graph_dispatcher() {
    // Best-effort drain so a short-lived context does not leave workers hanging.
    try {
        if (remote.has_open_dispatch()) {
            (void) remote.finish_dispatch();
        }
        (void) remote.drain_deferred();
    } catch (...) {
        // Destructor must not throw; workers will drop on socket close.
    }
    if (pred_thread_started_.load()) {
        {
            std::lock_guard<std::mutex> lock(pred_mutex_);
            pred_stop_ = true;
        }
        pred_cv_.notify_one();
        if (pred_thread_.joinable()) {
            pred_thread_.join();
        }
    }
    if (capture_file_ != nullptr) {
        std::fclose(capture_file_);
        capture_file_ = nullptr;
    }
    if (forward_log_ != nullptr) {
        std::fclose(forward_log_);
        forward_log_ = nullptr;
    }
    if (layer_trace_ != nullptr) {
        std::fclose(layer_trace_);
        layer_trace_ = nullptr;
    }
}

void graph_dispatcher::layer_trace_dense_begin(int32_t layer) noexcept {
    if (layer_trace_ == nullptr) {
        return;
    }
    layer_trace_record & record = layer_traces_[layer];
    record.dense_ns = 0;
    record.dense_started = dispatch_clock::now();
    record.dense_active = true;
}

void graph_dispatcher::layer_trace_dense_end(int32_t layer) noexcept {
    if (layer_trace_ == nullptr) {
        return;
    }
    const auto it = layer_traces_.find(layer);
    if (it == layer_traces_.end() || !it->second.dense_active) {
        return;
    }
    it->second.dense_ns += elapsed_ns(it->second.dense_started, dispatch_clock::now());
    it->second.dense_active = false;
}

void graph_dispatcher::write_layer_trace(int32_t layer) noexcept {
    if (layer_trace_ == nullptr) {
        return;
    }
    layer_trace_record & record = layer_traces_[layer];
    if (record.dense_active) {
        record.dense_ns += elapsed_ns(record.dense_started, dispatch_clock::now());
        record.dense_active = false;
    }
    const layer_trace_stats transport = remote.layer_trace(layer);
    std::fprintf(layer_trace_, "DS4 layer=%d dense_ns=%llu encode_ns=%llu send_ns=%llu recv_ns=%llu decode_ns=%llu scatter_ns=%llu\n",
                 layer,
                 (unsigned long long) record.dense_ns,
                 (unsigned long long) transport.encode_ns,
                 (unsigned long long) transport.send_ns,
                 (unsigned long long) transport.recv_ns,
                 (unsigned long long) transport.decode_ns,
                 0ull);
    std::fflush(layer_trace_);
}

ggml_tensor * graph_dispatcher::build(ggml_context * ctx,
                                      ggml_tensor *  activations,
                                      ggml_tensor *  selected_experts,
                                      ggml_tensor *  weights,
                                      int32_t        layer,
                                      float          swiglu_clamp) {
    if (layer < 0) {
        throw std::invalid_argument("expert dispatch requires a non-negative layer");
    }
    if (activations->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32 ||
        weights->type != GGML_TYPE_F32) {
        throw std::invalid_argument("expert dispatch requires F32 activations, I32 expert ids, and F32 weights");
    }

    auto & context = op_contexts[layer];
    if (!context) {
        context.reset(new op_context{ this, layer });
    }
    // Assign every call, not just on creation: op_contexts is cached per layer
    // for the lifetime of the dispatcher, so a create-only assignment would pin
    // whatever value the first graph build happened to see.
    context->swiglu_clamp = swiglu_clamp;
    return ggml_map_custom3(ctx, activations, selected_experts, weights, compute, 1, context.get());
}

ggml_tensor * graph_dispatcher::build_issue(ggml_context * ctx,
                                            ggml_tensor *  activations,
                                            ggml_tensor *  selected_experts,
                                            ggml_tensor *  weights,
                                            int32_t        layer,
                                            float          swiglu_clamp) {
    if (layer < 0) {
        throw std::invalid_argument("expert dispatch requires a non-negative layer");
    }
    if (activations->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32 ||
        weights->type != GGML_TYPE_F32) {
        throw std::invalid_argument("expert dispatch requires F32 activations, I32 expert ids, and F32 weights");
    }

    auto & context = op_contexts[layer];
    if (!context) {
        context.reset(new op_context{ this, layer });
    }
    context->swiglu_clamp = swiglu_clamp;
    ggml_tensor * issued =
        ggml_map_custom3(ctx, activations, selected_experts, weights, compute_issue, 1, context.get());
    context->issued = issued;
    return issued;
}

ggml_tensor * graph_dispatcher::after_issue(ggml_context * ctx, ggml_tensor * tensor, int32_t layer) {
    if (tensor == nullptr) {
        throw std::invalid_argument("after_issue requires a tensor");
    }
    const auto it = op_contexts.find(layer);
    if (it == op_contexts.end() || it->second == nullptr || it->second->issued == nullptr) {
        throw std::runtime_error("after_issue has no issue node for layer " + std::to_string(layer));
    }
    // 0 * issued[0], broadcast and added. Identity on finite values; the only
    // purpose is a compute edge so shexp cannot be scheduled before the send.
    ggml_tensor * gate = ggml_view_1d(ctx, it->second->issued, 1, 0);
    ggml_tensor * zero = ggml_scale(ctx, gate, 0.0f);
    return ggml_add(ctx, tensor, ggml_repeat(ctx, zero, tensor));
}

ggml_tensor * graph_dispatcher::build_wait(ggml_context * ctx, ggml_tensor * shexp, int32_t layer) {
    const auto it = op_contexts.find(layer);
    if (it == op_contexts.end() || it->second == nullptr || it->second->issued == nullptr) {
        throw std::runtime_error("build_wait has no issue node for layer " + std::to_string(layer));
    }
    if (shexp == nullptr) {
        throw std::invalid_argument("build_wait requires the shexp tensor so wait cannot overtake it");
    }
    return ggml_map_custom2(ctx, it->second->issued, shexp, compute_wait, 1, it->second.get());
}

size_t graph_dispatcher::n_workers() const {
    return remote.workers().size();
}

bool graph_dispatcher::hint_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("WP_PREFETCH_HINT");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

void graph_dispatcher::register_hash_layer(int32_t         layer,
                                           int32_t         n_expert_used,
                                           int32_t         n_vocab,
                                           const int32_t * data) {
    oracle_.register_layer(layer, n_expert_used, n_vocab, remote.n_expert(), data);
}

float graph_dispatcher::predicted_conf_min() {
    static const float value = [] {
        const char * v = std::getenv("WP_PREFETCH_CONF_MIN");
        if (v == nullptr || v[0] == '\0') {
            // Same 0.4 the draft head's own conf_min uses. A token the drafter
            // would not have kept is not worth a 9 MiB read either.
            return 0.4f;
        }
        const float f = strtof(v, nullptr);
        return f > 0.0f ? std::min(f, 1.0f) : 0.0f;   // 0 = gate off
    }();
    return value;
}

size_t graph_dispatcher::predicted_top_m() {
    static const size_t value = [] {
        const char * v = std::getenv("WP_PREFETCH_TOPM");
        if (v == nullptr || v[0] == '\0') {
            return (size_t) 6;   // n_expert_used: one token's worth of pages
        }
        const long m = strtol(v, nullptr, 10);
        return m > 0 ? (size_t) m : (size_t) 0;   // 0 = uncapped
    }();
    return value;
}

size_t graph_dispatcher::prefetch_for_tokens(const int32_t * tokens, size_t n_tokens,
                                             size_t n_certain, const float * conf) {
    if (!hint_enabled() || oracle_.empty() || tokens == nullptr || n_tokens == 0) {
        return 0;
    }
    if (n_certain > n_tokens) {
        n_certain = n_tokens;
    }
    size_t sent = 0;
    // Two passes, two frames, because a worker that receives one merged set
    // cannot tell which ids it may cheaply discard. The dedup below is per
    // (layer, provenance) for the same reason -- a predicted set that happens to
    // equal the last CERTAIN set is still a different statement.
    for (int pass = 0; pass < 2; ++pass) {
        const bool     certain    = (pass == 0);
        const size_t   beg        = certain ? 0 : n_certain;
        const size_t   count      = certain ? n_certain : n_tokens - n_certain;
        const uint32_t provenance = certain ? PIPE_HINT_CERTAIN : PIPE_HINT_PREDICTED;
        if (count == 0) {
            continue;
        }
        // CERTAIN tokens are ground truth, so their experts are certainties and
        // the flat union is exactly right. Only the PREDICTED tail carries a
        // confidence worth spending a fetch budget against.
        const float * pass_conf = (!certain && conf != nullptr) ? conf + beg : nullptr;
        for (int32_t layer : oracle_.layers()) {
            if (certain) {
                if (!oracle_.experts_for(layer, tokens + beg, count, hint_experts_) ||
                    hint_experts_.empty()) {
                    continue;
                }
            } else {
                if (!oracle_.experts_ranked(layer, tokens + beg, count, pass_conf, hint_ranked_) ||
                    hint_ranked_.empty()) {
                    continue;
                }

                // *** THE CONFIDENCE GATE. ***
                // Before this, a predicted frame was the UNION of every expert
                // every drafted token touches -- ~6 ids per token into a space of
                // 256, all indistinguishable. The worker then took the first M in
                // ASCENDING EXPERT ID, i.e. it selected by nothing. Both the gate
                // and the cap have to run HERE, because this is the only place
                // that knows how likely each id was.
                const float  conf_min = predicted_conf_min();
                const size_t top_m    = predicted_top_m();
                // *** THE FLOOR IS A PER-LAYER TRUST DECISION, NOT A PER-EXPERT FILTER. ***
                // Dropping individual experts below the floor has the same defect as
                // halving top-M: it leaves a layer partially covered, and a partially
                // covered layer pages in exactly like an uncovered one. So the floor
                // now asks "is this layer's prediction trustworthy at all" -- if the
                // BEST expert clears it, emit the whole set; if not, emit nothing and
                // spend no bandwidth. That still gates volume (an undecided layer
                // costs zero) without ever producing a useless partial set.
                if (conf_min > 0.0f) {
                    float best = 0.0f;
                    for (const hash_oracle::ranked_expert & e : hint_ranked_) {
                        best = std::max(best, e.conf);
                    }
                    if (best < conf_min) {
                        hint_ranked_.clear();
                    }
                }
                if (top_m != 0 && hint_ranked_.size() > top_m) {
                    // Partial sort by DESCENDING confidence, ties by ascending id
                    // so the surviving set is a deterministic function of the
                    // tokens -- last_hint_ dedup depends on that.
                    std::nth_element(hint_ranked_.begin(), hint_ranked_.begin() + (ptrdiff_t) top_m,
                                     hint_ranked_.end(),
                                     [](const hash_oracle::ranked_expert & a,
                                        const hash_oracle::ranked_expert & b) {
                                         return a.conf != b.conf ? a.conf > b.conf
                                                                 : a.expert_id < b.expert_id;
                                     });
                    hint_ranked_.resize(top_m);
                }
                if (hint_ranked_.empty()) {
                    continue;
                }

                // Back to ascending expert id: that is the wire's dedup invariant
                // (pipe_encode_expert_prefetch_hint rejects anything else), and
                // the worker no longer needs the ranking -- the selection it used
                // to approximate has already happened.
                hint_experts_.clear();
                hint_experts_.reserve(hint_ranked_.size());
                for (const hash_oracle::ranked_expert & e : hint_ranked_) {
                    hint_experts_.push_back(e.expert_id);
                }
                std::sort(hint_experts_.begin(), hint_experts_.end());
            }
            // Same set as last time for this layer and provenance: the worker
            // would resolve it to pages it already holds and discard it. Skip.
            std::vector<int32_t> & previous = last_hint_[layer * 2 + pass];
            if (previous == hint_experts_) {
                continue;
            }
            previous = hint_experts_;
            // Swallowing here is deliberate and is the property that makes this
            // safe to leave enabled: a hint carries no correctness weight, so no
            // failure inside it may reach the decode. A broken socket surfaces on
            // the next real dispatch, which is where it belongs.
            try {
                sent += remote.send_prefetch_hints(layer, hint_experts_, provenance, (uint32_t) count);
            } catch (...) {
                return sent;
            }
        }
    }
    return sent;
}

int graph_dispatcher::router2_topm() {
    static const int value = [] {
        const char * v = std::getenv("WP_HINT_ROUTER2");
        if (v == nullptr || v[0] == '\0') {
            return 0;
        }
        return std::min<int32_t>(PREFETCH_HINT_MAX_EXPERTS, std::max(0, std::atoi(v)));
    }();
    return value;
}

float graph_dispatcher::router2_conf_min() {
    static const float value = [] {
        const char * v = std::getenv("WP_HINT_ROUTER2_CONF");
        if (v == nullptr || v[0] == '\0') {
            return 0.10f;   // the whole-expert pager's WP_HOST_PREFETCH_MIN_CONF default
        }
        const float f = strtof(v, nullptr);
        return f > 0.0f ? std::min(f, 1.0f) : 0.0f;   // 0 = gate off, for the A/B
    }();
    return value;
}

// WP_HINT_ROUTER2_DEPTH_DECAY=1 restores the halving of top-M with depth.
// DEFAULT OFF -- see the note at its use site: emitting fewer experts than a
// layer consumes cannot stop that layer paging in.
static const bool s_depth_decay = [] {
    const char * v = std::getenv("WP_HINT_ROUTER2_DEPTH_DECAY");
    return v != nullptr && v[0] == '1';
}();

int graph_dispatcher::router2_lookahead() {
    static const int value = [] {
        const char * v = std::getenv("WP_HINT_ROUTER2_K");
        if (v == nullptr || v[0] == '\0') {
            return 1;
        }
        // Clamped only to the model's layer count -- 8 was a number I picked,
        // and it was cutting off exactly the horizon that makes prefetch work.
        // Lead time IS the mechanism: at K=1 the target is ~10 ms out, which is
        // inside the window where the demand path already holds the page, so
        // every candidate skipped as VRAM-resident. K=7 produced the first host
        // landings this rig has ever recorded. Precision decays with distance,
        // which is what the top-M halving and the rising floor are for.
        return std::min(128, std::max(1, std::atoi(v)));
    }();
    return value;
}

float graph_dispatcher::router2_conf_step() {
    static const float value = [] {
        const char * v = std::getenv("WP_HINT_ROUTER2_CONF_STEP");
        if (v == nullptr || v[0] == '\0') {
            return 0.05f;
        }
        const float f = strtof(v, nullptr);
        return f > 0.0f ? f : 0.0f;
    }();
    return value;
}

int graph_dispatcher::predict_max_tokens() {
    static const int value = [] {
        const char * v = std::getenv("WP_PREDICT_MAX_TOKENS");
        if (v == nullptr || v[0] == '\0') {
            return 16;
        }
        return std::max(1, std::atoi(v));
    }();
    return value;
}

void graph_dispatcher::register_router_layer(int32_t layer, int32_t n_expert, int32_t n_embd,
                                             const float * w, const float * b) {
    if (n_expert != remote.n_expert() || n_embd != remote.n_embd()) {
        throw std::invalid_argument("router layer dims do not match the dispatcher");
    }
    if (w == nullptr || b == nullptr) {
        throw std::invalid_argument("router layer registered without weights");
    }
    router_layer & rl = routers_[layer];
    rl.w.assign(w, w + (size_t) n_expert * (size_t) n_embd);
    rl.b.assign(b, b + (size_t) n_expert);
}

void graph_dispatcher::enqueue_prediction(int32_t layer, const std::vector<float> & activations,
                                          int64_t n_tokens) noexcept {
    try {
        if (router2_topm() <= 0 || n_tokens <= 0 ||
            n_tokens > (int64_t) PREFETCH_HINT_MAX_TOKENS || routers_.empty()) {
            return;
        }
        bool any_target = false;
        for (int d = 0; d < router2_lookahead(); ++d) {
            if (routers_.find(layer + 2 + d) != routers_.end()) {
                any_target = true;
                break;
            }
        }
        if (!any_target) {
            return;   // past the last layer, or the oracle was cleared
        }
        if (!pred_thread_started_.exchange(true)) {
            pred_thread_ = std::thread([this] { predictor_loop(); });
        }
        {
            std::lock_guard<std::mutex> lock(pred_mutex_);
            // *** LATEST-WINS MAILBOX: COUNT WHAT IT THROWS AWAY. ***
            // This is a ONE-SLOT handoff. enqueue_prediction is called once per
            // LAYER (43x per token); if the scorer thread is still working on
            // the previous snapshot, this overwrite silently discards a whole
            // layer's prediction. The scorer does an n_expert x n_embd GEMV per
            // target layer per snapshot, so at 43 layers/token it plausibly
            // cannot keep up -- meaning the effective prediction rate, and
            // WHICH layers get predicted, are decided by thread timing rather
            // than by anything principled. Until 2026-08-19 nothing recorded
            // that, so every conclusion about K and conf was drawn without
            // knowing how often the predictor actually ran.
            if (pred_inbox_.valid) {
                ++pred_dropped_;
            }
            ++pred_offered_;
            pred_inbox_.layer    = layer;
            pred_inbox_.n_tokens = n_tokens;
            pred_inbox_.activations.assign(activations.begin(), activations.end());
            pred_inbox_.valid    = true;
        }
        pred_cv_.notify_one();
    } catch (...) {
        // Swallow. A dropped snapshot costs one hint, never the decode.
    }
}

void graph_dispatcher::predictor_loop() {
    for (;;) {
        pred_job job;
        {
            std::unique_lock<std::mutex> lock(pred_mutex_);
            pred_cv_.wait(lock, [this] { return pred_stop_ || pred_inbox_.valid; });
            if (pred_stop_) {
                return;
            }
            job = std::move(pred_inbox_);
            pred_inbox_.valid = false;
            ++pred_scored_;
        }
        try {
            const int32_t n_expert = remote.n_expert();
            const int32_t n_embd   = remote.n_embd();
            const int     K        = router2_lookahead();
            const int     base_m   = router2_topm();
            // DECAYING HORIZON, same policy as the whole-expert pager's
            // submit_xlayer_prefetch: the nearest target gets the full top-M at
            // the base floor, and each layer further out gets HALF the width and
            // a HIGHER floor. Prediction quality falls with distance, so a fixed
            // (M, conf) at depth 3 spends real bandwidth on noise -- and on this
            // rig speculative bytes compete with demand reads for the same queue.
            for (int d = 0; d < K; ++d) {
                const int32_t target = job.layer + 2 + d;
                const auto    it     = routers_.find(target);
                if (it == routers_.end()) {
                    continue;
                }
                // *** M IS NEVER LESS THAN n_expert_used. ***
                // This used to halve top-M with depth (base_m >> d), ported from
                // the whole-expert pager's submit_xlayer_prefetch. That is wrong
                // here and it is not a tuning knob, it is a defect: a layer routes
                // to n_expert_used experts, so prefetching FEWER than that leaves
                // the layer paging in anyway. The read is spent and the stall
                // still happens. Partial coverage of a layer is worth nothing --
                // coverage is all-or-nothing per layer.
                // At K=36 the old rule emitted 6,3,1,1,1... i.e. one expert against
                // a layer needing six, which cannot prevent a single page-in.
                // WP_HINT_ROUTER2_DEPTH_DECAY=1 restores the old behaviour for A/B.
                const int32_t m = s_depth_decay ? std::max(1, base_m >> d) : base_m;
                // Clamp at 1.0 -- the bound of a PROBABILITY -- not at an
                // arbitrary 0.99. The whole-expert pager hardcodes 0.99 here
                // (wp-pager.cpp:838 and :960) and that ceiling silently
                // overrides an explicit setting: ask for a 0.995 floor and you
                // get 0.99 with nothing reporting the difference. A floor that
                // reaches 1.0 at depth d means "emit nothing that far out",
                // which is a legitimate answer and needs no magic number.
                const float   conf =
                    std::min(1.0f, router2_conf_min() + (float) d * router2_conf_step());
                const router_layer & rl = it->second;
                std::vector<int32_t> experts = router2_top_experts(
                    rl.w.data(), rl.b.data(), job.activations.data(), job.n_tokens,
                    n_expert, n_embd, m, conf);
                if (experts.empty()) {
                    continue;   // the gate rejected the whole layer: correct, not a failure
                }
                {
                    std::lock_guard<std::mutex> lock(pred_mutex_);
                    pred_ready_[target] = { (uint32_t) job.n_tokens, std::move(experts) };
                }
            }
        } catch (...) {
            // Swallow and keep serving; one lost prediction is one lost hint.
        }
    }
}

void graph_dispatcher::flush_predicted_hints() noexcept {
    try {
        std::map<int32_t, pred_result> ready;
        {
            std::lock_guard<std::mutex> lock(pred_mutex_);
            if (pred_ready_.empty()) {
                return;
            }
            ready.swap(pred_ready_);
        }
        for (auto & entry : ready) {
            std::vector<int32_t> & previous = last_pred_hint_[entry.first];
            if (previous == entry.second.experts) {
                continue;   // the worker would resolve it to pages already queued
            }
            previous = entry.second.experts;
            (void) remote.send_prefetch_hints(entry.first, entry.second.experts,
                                              PIPE_HINT_PREDICTED, entry.second.n_tokens);
        }
    } catch (...) {
        // Swallow. A broken socket surfaces on the next real dispatch.
    }
}

size_t graph_dispatcher::prefetch_ngram_for_tokens(const int32_t * tokens, size_t n_tokens) noexcept {
    if (ngram_table_ == nullptr || tokens == nullptr || n_tokens == 0 ||
        n_tokens > PREFETCH_HINT_MAX_TOKENS) {
        return 0;
    }
    size_t sent = 0;
    try {
        for (int32_t layer = 0; layer < ngram_table_->n_layers(); ++layer) {
            std::vector<int32_t> experts =
                ngram_table_->top_experts(tokens, n_tokens, layer, ngram_top_m_);
            if (experts.empty()) {
                continue;
            }
            std::vector<int32_t> & previous = last_ngram_hint_[layer];
            if (previous == experts) {
                continue;
            }
            previous = experts;
            sent += remote.send_prefetch_hints(layer, experts, PIPE_HINT_PREDICTED,
                                               (uint32_t) n_tokens);
        }
    } catch (...) {
    }
    return sent;
}

void graph_dispatcher::capture_routing(const char * prefix, int32_t layer,
                                       const std::vector<float> & activations,
                                       const ggml_tensor * selected_experts,
                                       int64_t n_tokens, int64_t n_expert_used) noexcept {
    try {
        if (n_tokens <= 0 || n_tokens > predict_max_tokens()) {
            return;
        }
        if (capture_file_ == nullptr) {
            const std::string path =
                std::string(prefix) + "." + std::to_string((long) getpid()) + ".bin";
            capture_file_ = std::fopen(path.c_str(), "ab");
            if (capture_file_ == nullptr) {
                return;
            }
        }
        const int32_t n_embd = remote.n_embd();
        // Record: magic 'WPC1', layer, n_tokens, n_embd, n_expert_used,
        // f32 h[n_tokens * n_embd], i32 sel[n_tokens * n_expert_used].
        const uint32_t magic  = 0x31435057;
        const int32_t  header[4] = { layer, (int32_t) n_tokens, n_embd, (int32_t) n_expert_used };
        std::fwrite(&magic, sizeof(magic), 1, capture_file_);
        std::fwrite(header, sizeof(header), 1, capture_file_);
        std::fwrite(activations.data(), sizeof(float),
                    (size_t) n_tokens * (size_t) n_embd, capture_file_);
        for (int64_t i = 0; i < n_tokens * n_expert_used; ++i) {
            const int32_t expert = ggml_get_i32_1d(selected_experts, (int) i);
            std::fwrite(&expert, sizeof(expert), 1, capture_file_);
        }
    } catch (...) {
        // Capture is diagnostics; losing a record must not touch the decode.
    }
}

void graph_dispatcher::note_batch_tokens(const int32_t * tokens, size_t n_tokens) noexcept {
    try {
        if (tokens == nullptr || n_tokens == 0) {
            return;
        }
        if (capture_file_ == nullptr) {
            const char * prefix = std::getenv("WP_PREDICT_CAPTURE");
            if (prefix == nullptr || prefix[0] == '\0') {
                return;
            }
            const std::string path = std::string(prefix) + "." +
                std::to_string((long) getpid()) + ".bin";
            capture_file_ = std::fopen(path.c_str(), "ab");
            if (capture_file_ == nullptr) {
                return;
            }
        }
        const uint32_t marker = 0x31425457u;
        const uint32_t count = (uint32_t) n_tokens;
        std::fwrite(&marker, sizeof(marker), 1, capture_file_);
        std::fwrite(&count, sizeof(count), 1, capture_file_);
        std::fwrite(tokens, sizeof(int32_t), n_tokens, capture_file_);
        std::fflush(capture_file_);
    } catch (...) {
    }
}

void graph_dispatcher::latch_failure(const char * message) noexcept {
    try {
        std::lock_guard<std::mutex> lock(failure_mutex_);
        if (failed_.load(std::memory_order_relaxed)) {
            return;
        }
        failure_message_ = message;
        failed_.store(true, std::memory_order_release);
    } catch (...) {
        failed_.store(true, std::memory_order_release);
    }
}

bool graph_dispatcher::failed() const noexcept {
    return failed_.load(std::memory_order_acquire);
}

std::string graph_dispatcher::failure_message() const {
    std::lock_guard<std::mutex> lock(failure_mutex_);
    return failure_message_;
}

void graph_dispatcher::begin_decode() noexcept {
    remote.begin_deferral_window();
    decode_t0_ = dispatch_clock::now();
    if (!collect_stats_ && forward_log_ == nullptr) {
        return;
    }
    decode_active_                       = true;
    decode_layers_                       = 0;
    decode_ns_pack_                      = 0;
    decode_ns_issue_                     = 0;
    decode_ns_wait_                      = 0;
    decode_ns_unpack_                    = 0;
    decode_ns_total_                     = 0;
    decode_first_await_in_flight_        = 0;
    decode_max_in_flight_                = 0;
    decode_n_tokens_                     = 0;
    decode_workers_.clear();
    for (const worker_info & worker : remote.workers()) {
        decode_workers_.push_back({ worker.endpoint });
    }
}

void graph_dispatcher::end_decode() noexcept {
    // PREDICTION CADENCE. Emitted once per decode so a run's log says how often
    // the predictor actually ran, rather than how often it was asked to. See the
    // latest-wins comment in enqueue_prediction.
    if (router2_topm() > 0) {
        const uint64_t offered = pred_offered();
        const uint64_t dropped = pred_dropped();
        if (offered > 0) {
            // stderr, not LLAMA_LOG_INFO: the library log level filters INFO out of
            // the router's journal (verified 2026-08-19 -- the same reason the
            // union diagnostic uses fprintf).
            std::fprintf(stderr,
                "expert dispatch predictor: offered=%llu scored=%llu dropped=%llu (%.1f%% dropped)\n",
                (unsigned long long) offered,
                (unsigned long long) pred_scored(),
                (unsigned long long) dropped,
                100.0 * (double) dropped / (double) offered);
        }
    }
    // Last-layer path should have left nothing pending; if anything remains it
    // missed its fold point (bug). Drain and count as late inside the dispatcher.
    try {
        const std::vector<float> leftover = remote.drain_deferred();
        if (!leftover.empty()) {
            LLAMA_LOG_WARN(
                "expert dispatch: drained %zu deferred partial values after decode "
                "(n_deferred_late will reflect this; last layer must not defer)\n",
                leftover.size());
        }
    } catch (const std::exception & error) {
        latch_failure(error.what());
    } catch (...) {
        latch_failure("expert dispatch failed while draining deferred partials");
    }

    remote.end_deferral_window();

    const deferral_stats & dstats = remote.get_deferral_stats();
    // Always log mechanism counters at WARN so they are visible under the
    // llama-server default logger threshold (3); INFO is filtered.
    LLAMA_LOG_WARN(
        "expert dispatch deferral: WP_DEFER_K=%d nvme_util_pct=%.1f ns_gap=%.2f ms "
        "n_deferred=%llu n_deferred_late=%llu\n",
        dstats.defer_k,
        dstats.nvme_util_pct,
        dstats.ns_gap * 1.0e-6,
        (unsigned long long) dstats.n_deferred,
        (unsigned long long) dstats.n_deferred_late);

    // Same treatment for the prefetch hint, and for the same reason: these are
    // the spine's side of the mechanism, and they are what tells a run that
    // "prefetch did nothing" apart from "prefetch was never offered". Silent
    // when the feature is off, so a config-of-record run's log is unchanged.
    if (hint_enabled() || router2_topm() > 0 || ngram_table_ != nullptr) {
        const prefetch_hint_stats & hstats = remote.get_prefetch_hint_stats();
        LLAMA_LOG_WARN(
            "expert dispatch prefetch hint: layers=%zu frames=%llu experts=%llu "
            "send_failed=%llu no_route=%llu skip_dynamic=%llu skip_in_flight=%llu\n",
            oracle_.layers().size(),
            (unsigned long long) hstats.n_frames,
            (unsigned long long) hstats.n_experts,
            (unsigned long long) hstats.n_send_failed,
            (unsigned long long) hstats.n_no_oracle,
            (unsigned long long) hstats.n_skipped_dynamic,
            (unsigned long long) hstats.n_skipped_in_flight);
    }

    const uint64_t ns_wall = elapsed_ns(decode_t0_, dispatch_clock::now());
    if (forward_log_ != nullptr && decode_active_ && decode_layers_ > 0) {
        const uint64_t ns_dispatch = decode_ns_total_;
        const uint64_t ns_other = ns_wall > ns_dispatch ? ns_wall - ns_dispatch : 0;
        const double epoch_end =
            (double) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() / 1e6;
        std::fprintf(forward_log_,
                     "%u %zu %llu %llu %llu %llu %llu %llu %llu %.6f\n",
                     decode_n_tokens_,
                     decode_layers_,
                     (unsigned long long) ns_wall,
                     (unsigned long long) decode_ns_pack_,
                     (unsigned long long) decode_ns_issue_,
                     (unsigned long long) decode_ns_wait_,
                     (unsigned long long) decode_ns_unpack_,
                     (unsigned long long) ns_dispatch,
                     (unsigned long long) ns_other,
                     epoch_end);
        std::fflush(forward_log_);
    }
    if (!collect_stats_ || !decode_active_) {
        decode_active_ = false;
        return;
    }
    decode_active_ = false;
    if (decode_layers_ == 0) {
        return;
    }

    const double ns_to_ms = 1.0e-6;
    // phase= and n_tokens= (2026-08-03): this block is emitted once per
    // llama_context::decode(), so prefill and decode blocks were previously
    // distinguishable only by their position in the log. With a real prompt and
    // n_ubatch chunking there are several prefill blocks, and any tooling that
    // assumed "block 1 is prefill" silently mis-attributed the rest.
    // ms/tok is the cross-phase-comparable column: prefill amortises, decode does not.
    LLAMA_LOG_WARN(
        "expert dispatch: phase=%s n_tokens=%u layers=%zu pack=%.2f ms issue=%.2f ms wait=%.2f ms "
        "unpack=%.2f ms total=%.2f ms (%.3f ms/tok) first_await_in_flight avg=%.1f max_in_flight=%zu "
        "(workers=%zu)\n",
        // 3-way: n_tokens > 1 alone would call every speculative-verify batch a
        // prefill, and with DSpark on that is ~97% of them by count.
        decode_n_tokens_ >= 64 ? "PREFILL" : (decode_n_tokens_ > 1 ? "verify" : "decode"),
        decode_n_tokens_,
        decode_layers_,
        decode_ns_pack_ * ns_to_ms,
        decode_ns_issue_ * ns_to_ms,
        decode_ns_wait_ * ns_to_ms,
        decode_ns_unpack_ * ns_to_ms,
        decode_ns_total_ * ns_to_ms,
        decode_ns_total_ * ns_to_ms / (double) std::max<uint32_t>(1, decode_n_tokens_),
        (double) decode_first_await_in_flight_ / (double) decode_layers_,
        decode_max_in_flight_,
        n_workers());
    for (const worker_dispatch_stats & worker : decode_workers_) {
        const double avg_wait_ms = worker.n_requests == 0
            ? 0.0
            : worker.ns_wait * ns_to_ms / (double) worker.n_requests;
        LLAMA_LOG_WARN(
            "expert dispatch worker %s requests=%llu experts=%llu wait=%.2f ms (avg %.2f ms/req)\n",
            worker.endpoint.c_str(),
            (unsigned long long) worker.n_requests,
            (unsigned long long) worker.n_experts_total,
            worker.ns_wait * ns_to_ms,
            avg_wait_ms);
    }
}

void graph_dispatcher::compute(ggml_tensor *       dst,
                               const ggml_tensor * activations,
                               const ggml_tensor * selected_experts,
                               const ggml_tensor * weights,
                               int                 ith,
                               int                 nth,
                               void *              userdata) {
    graph_dispatcher * owner = nullptr;
    try {
        if (ith != 0 || nth != 1) {
            throw std::runtime_error("expert dispatch custom op must run as one CPU task");
        }
        op_context * context = static_cast<op_context *>(userdata);
        if (context == nullptr || context->owner == nullptr) {
            throw std::runtime_error("expert dispatch custom op has no dispatcher");
        }
        owner = context->owner;
        if (owner->failed()) {
            ggml_set_zero(dst);
            return;
        }
        const bool collect_stats = owner->decode_active_;
        const dispatch_clock::time_point total_start =
            collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};

        const int64_t n_embd        = activations->ne[0];
        const int64_t n_tokens      = activations->ne[1];
        const int64_t n_expert_used = selected_experts->ne[0];
        // Weights must be [1, n_expert_used, n_tokens] — a bare [n_tokens] 1-D
        // tensor PASSES ggml_can_repeat against [n_embd, n_tokens] (because
        // ne[i] % src1->ne[i] == 0) and then broadcasts along the WRONG AXIS.
        const bool shapes_match =
            n_embd == owner->remote.n_embd() && n_tokens > 0 &&
            activations->ne[2] == 1 && activations->ne[3] == 1 &&
            selected_experts->ne[1] == n_tokens &&
            selected_experts->ne[2] == 1 && selected_experts->ne[3] == 1 &&
            weights->ne[0] == 1 && weights->ne[1] == n_expert_used &&
            weights->ne[2] == n_tokens && weights->ne[3] == 1 &&
            ggml_are_same_shape(dst, activations);
        if (!shapes_match) {
            throw std::runtime_error("expert dispatch custom op input shapes do not match");
        }

        // f32 STRAIGHT THROUGH as of PIPE_VERSION 4. This used to round to f16
        // here, which put a ~3e-4 relative error on the input of EVERY routed
        // expert on EVERY layer while attention, the shared expert and the
        // residual all kept f32. See pipe_expert_dispatch_req::activations.
        std::vector<float> wire_activations((size_t) n_embd * (size_t) n_tokens);
        const dispatch_clock::time_point pack_start =
            collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};
        if (ggml_is_contiguous(activations)) {
            std::memcpy(wire_activations.data(), activations->data,
                        wire_activations.size() * sizeof(float));
        } else {
            for (size_t i = 0; i < wire_activations.size(); ++i) {
                wire_activations[i] = ggml_get_f32_1d(activations, (int) i);
            }
        }
        const dispatch_clock::time_point pack_end =
            collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};

        std::map<int32_t, pipe_expert_assignment> by_expert;
        for (int64_t token = 0; token < n_tokens; ++token) {
            std::set<int32_t> token_experts;
            for (int64_t slot = 0; slot < n_expert_used; ++slot) {
                const int     index  = (int) (token * n_expert_used + slot);
                const int32_t expert = ggml_get_i32_1d(selected_experts, index);
                if (!token_experts.insert(expert).second) {
                    throw std::runtime_error("expert dispatch received a repeated expert for one token");
                }
                auto inserted = by_expert.emplace(expert, pipe_expert_assignment{});
                if (inserted.second) {
                    inserted.first->second.expert_id = expert;
                    inserted.first->second.weights.resize((size_t) n_tokens, 0.0f);
                }
                inserted.first->second.weights[(size_t) token] = ggml_get_f32_1d(weights, index);
            }
        }

        std::vector<pipe_expert_assignment> assignments;
        assignments.reserve(by_expert.size());
        for (auto & entry : by_expert) {
            assignments.push_back(std::move(entry.second));
        }

        // Predicted hints: flush whatever the scorer thread finished (sockets
        // are quiet here -- in_flight == 0 under WP_DEFER_K=0; with deferral
        // on, send_prefetch_hints declines and counts it), then hand it this
        // layer's activations. The GEMM runs off-thread and its result ships
        // at the NEXT layer's entry -- one layer of lead spent on the handoff
        // instead of +26 ms/step of critical-path scoring (2026-08-07 A/B).
        if (owner->router2_topm() > 0) {
            owner->flush_predicted_hints();
            owner->enqueue_prediction(context->layer, wire_activations, n_tokens);
        }
        if (const char * capture_prefix = std::getenv("WP_PREDICT_CAPTURE");
            capture_prefix != nullptr && capture_prefix[0] != '\0') {
            owner->capture_routing(capture_prefix, context->layer, wire_activations,
                                   selected_experts, n_tokens, n_expert_used);
        }

        const uint64_t           seq_id = owner->next_seq_id.fetch_add(1, std::memory_order_relaxed);
        // dispatch() issues deferred reads before returning, waits only for
        // immediate experts, and folds the previous layer's deferred partials
        // into the returned block (residual path for layer N+1).
        const std::vector<float> result =
            owner->remote.dispatch(context->layer, seq_id, (uint32_t) n_tokens, wire_activations, assignments,
                                   context->swiglu_clamp);
        const dispatch_stats & layer_stats = owner->remote.last_dispatch_stats();
        const dispatch_clock::time_point unpack_start =
            collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};
        if (ggml_is_contiguous(dst) && dst->type == GGML_TYPE_F32) {
            std::memcpy(dst->data, result.data(), result.size() * sizeof(float));
        } else {
            for (size_t i = 0; i < result.size(); ++i) {
                ggml_set_f32_1d(dst, (int) i, result[i]);
            }
        }

        if (collect_stats) {
            const dispatch_clock::time_point total_end = dispatch_clock::now();
            ++owner->decode_layers_;
            owner->decode_n_tokens_ = (uint32_t) n_tokens;
            owner->decode_ns_pack_ += elapsed_ns(pack_start, pack_end);
            owner->decode_ns_issue_ += layer_stats.ns_issue;
            owner->decode_ns_wait_ += layer_stats.ns_wait;
            owner->decode_ns_unpack_ += elapsed_ns(unpack_start, total_end);
            owner->decode_ns_total_ += elapsed_ns(total_start, total_end);
            owner->decode_first_await_in_flight_ += layer_stats.first_await_in_flight;
            owner->decode_max_in_flight_ = std::max(owner->decode_max_in_flight_, layer_stats.max_in_flight);
            for (const worker_dispatch_stats & worker : layer_stats.workers) {
                for (worker_dispatch_stats & decode_worker : owner->decode_workers_) {
                    if (decode_worker.endpoint != worker.endpoint) {
                        continue;
                    }
                    decode_worker.ns_wait += worker.ns_wait;
                    decode_worker.n_requests += worker.n_requests;
                    decode_worker.n_experts_total += worker.n_experts_total;
                    break;
                }
            }
        }
    } catch (const std::exception & error) {
        if (owner != nullptr) {
            owner->latch_failure(error.what());
        }
        ggml_set_zero(dst);
    } catch (...) {
        if (owner != nullptr) {
            owner->latch_failure("expert dispatch custom op failed with an unknown exception");
        }
        ggml_set_zero(dst);
    }
}

void graph_dispatcher::compute_issue(ggml_tensor *       dst,
                                     const ggml_tensor * activations,
                                     const ggml_tensor * selected_experts,
                                     const ggml_tensor * weights,
                                     int                 ith,
                                     int                 nth,
                                     void *              userdata) {
    graph_dispatcher * owner = nullptr;
    try {
        if (ith != 0 || nth != 1) {
            throw std::runtime_error("expert dispatch custom op must run as one CPU task");
        }
        op_context * context = static_cast<op_context *>(userdata);
        if (context == nullptr || context->owner == nullptr) {
            throw std::runtime_error("expert dispatch custom op has no dispatcher");
        }
        owner = context->owner;
        if (owner->failed()) {
            if (dst->data != nullptr && dst->type == GGML_TYPE_F32 && ggml_nelements(dst) > 0) {
                *static_cast<float *>(dst->data) = 0.0f;
            }
            return;
        }
        const bool collect_stats = owner->decode_active_;

        const int64_t n_embd        = activations->ne[0];
        const int64_t n_tokens      = activations->ne[1];
        const int64_t n_expert_used = selected_experts->ne[0];
        const bool shapes_match =
            n_embd == owner->remote.n_embd() && n_tokens > 0 &&
            activations->ne[2] == 1 && activations->ne[3] == 1 &&
            selected_experts->ne[1] == n_tokens &&
            selected_experts->ne[2] == 1 && selected_experts->ne[3] == 1 &&
            weights->ne[0] == 1 && weights->ne[1] == n_expert_used &&
            weights->ne[2] == n_tokens && weights->ne[3] == 1;
        if (!shapes_match) {
            throw std::runtime_error("expert dispatch issue op input shapes do not match");
        }

        std::vector<float> wire_activations((size_t) n_embd * (size_t) n_tokens);
        const dispatch_clock::time_point pack_start =
            collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};
        if (ggml_is_contiguous(activations)) {
            std::memcpy(wire_activations.data(), activations->data,
                        wire_activations.size() * sizeof(float));
        } else {
            for (size_t i = 0; i < wire_activations.size(); ++i) {
                wire_activations[i] = ggml_get_f32_1d(activations, (int) i);
            }
        }
        const dispatch_clock::time_point pack_end =
            collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};

        std::map<int32_t, pipe_expert_assignment> by_expert;
        for (int64_t token = 0; token < n_tokens; ++token) {
            std::set<int32_t> token_experts;
            for (int64_t slot = 0; slot < n_expert_used; ++slot) {
                const int     index  = (int) (token * n_expert_used + slot);
                const int32_t expert = ggml_get_i32_1d(selected_experts, index);
                if (!token_experts.insert(expert).second) {
                    throw std::runtime_error("expert dispatch received a repeated expert for one token");
                }
                auto inserted = by_expert.emplace(expert, pipe_expert_assignment{});
                if (inserted.second) {
                    inserted.first->second.expert_id = expert;
                    inserted.first->second.weights.resize((size_t) n_tokens, 0.0f);
                }
                inserted.first->second.weights[(size_t) token] = ggml_get_f32_1d(weights, index);
            }
        }

        std::vector<pipe_expert_assignment> assignments;
        assignments.reserve(by_expert.size());
        for (auto & entry : by_expert) {
            assignments.push_back(std::move(entry.second));
        }

        if (owner->router2_topm() > 0) {
            owner->flush_predicted_hints();
            owner->enqueue_prediction(context->layer, wire_activations, n_tokens);
        }
        if (const char * capture_prefix = std::getenv("WP_PREDICT_CAPTURE");
            capture_prefix != nullptr && capture_prefix[0] != '\0') {
            owner->capture_routing(capture_prefix, context->layer, wire_activations,
                                   selected_experts, n_tokens, n_expert_used);
        }

        const uint64_t seq_id = owner->next_seq_id.fetch_add(1, std::memory_order_relaxed);
        owner->remote.begin_dispatch(context->layer, seq_id, (uint32_t) n_tokens, wire_activations, assignments,
                                     context->swiglu_clamp);
        if (dst->data != nullptr && dst->type == GGML_TYPE_F32 && ggml_nelements(dst) > 0) {
            *static_cast<float *>(dst->data) = 0.0f;
        }
        if (collect_stats) {
            const dispatch_stats & layer_stats = owner->remote.last_dispatch_stats();
            owner->decode_n_tokens_ = (uint32_t) n_tokens;
            owner->decode_ns_pack_ += elapsed_ns(pack_start, pack_end);
            owner->decode_ns_issue_ += layer_stats.ns_issue;
        }
    } catch (const std::exception & error) {
        if (owner != nullptr) {
            owner->latch_failure(error.what());
        }
        if (dst->data != nullptr && dst->type == GGML_TYPE_F32 && ggml_nelements(dst) > 0) {
            *static_cast<float *>(dst->data) = 0.0f;
        }
    } catch (...) {
        if (owner != nullptr) {
            owner->latch_failure("expert dispatch issue op failed with an unknown exception");
        }
        if (dst->data != nullptr && dst->type == GGML_TYPE_F32 && ggml_nelements(dst) > 0) {
            *static_cast<float *>(dst->data) = 0.0f;
        }
    }
}

void graph_dispatcher::compute_wait(ggml_tensor *       dst,
                                    const ggml_tensor * issued,
                                    const ggml_tensor * shexp,
                                    int                 ith,
                                    int                 nth,
                                    void *              userdata) {
    GGML_UNUSED(issued);
    GGML_UNUSED(shexp);
    graph_dispatcher * owner = nullptr;
    try {
        if (ith != 0 || nth != 1) {
            throw std::runtime_error("expert dispatch custom op must run as one CPU task");
        }
        op_context * context = static_cast<op_context *>(userdata);
        if (context == nullptr || context->owner == nullptr) {
            throw std::runtime_error("expert dispatch wait op has no dispatcher");
        }
        owner = context->owner;
        if (owner->failed()) {
            ggml_set_zero(dst);
            return;
        }
        const bool collect_stats = owner->decode_active_;
        const dispatch_clock::time_point unpack_start =
            collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};
        const std::vector<float> result = owner->remote.finish_dispatch();
        const dispatch_stats & layer_stats = owner->remote.last_dispatch_stats();
        if (ggml_is_contiguous(dst) && dst->type == GGML_TYPE_F32) {
            std::memcpy(dst->data, result.data(), result.size() * sizeof(float));
        } else {
            for (size_t i = 0; i < result.size(); ++i) {
                ggml_set_f32_1d(dst, (int) i, result[i]);
            }
        }
        owner->write_layer_trace(context->layer);
        if (collect_stats) {
            const dispatch_clock::time_point total_end = dispatch_clock::now();
            ++owner->decode_layers_;
            owner->decode_ns_wait_ += layer_stats.ns_wait;
            owner->decode_ns_unpack_ += elapsed_ns(unpack_start, total_end);
            owner->decode_ns_total_ += layer_stats.ns_issue + layer_stats.ns_wait +
                                       elapsed_ns(unpack_start, total_end);
            owner->decode_first_await_in_flight_ += layer_stats.first_await_in_flight;
            owner->decode_max_in_flight_ = std::max(owner->decode_max_in_flight_, layer_stats.max_in_flight);
            for (const worker_dispatch_stats & worker : layer_stats.workers) {
                for (worker_dispatch_stats & decode_worker : owner->decode_workers_) {
                    if (decode_worker.endpoint != worker.endpoint) {
                        continue;
                    }
                    decode_worker.ns_wait += worker.ns_wait;
                    decode_worker.n_requests += worker.n_requests;
                    decode_worker.n_experts_total += worker.n_experts_total;
                    break;
                }
            }
        }
    } catch (const std::exception & error) {
        if (owner != nullptr) {
            owner->latch_failure(error.what());
        }
        ggml_set_zero(dst);
    } catch (...) {
        if (owner != nullptr) {
            owner->latch_failure("expert dispatch wait op failed with an unknown exception");
        }
        ggml_set_zero(dst);
    }
}

}  // namespace pipe_expert_dispatcher
