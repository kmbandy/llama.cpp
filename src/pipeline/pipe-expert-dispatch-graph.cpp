#include "pipe-expert-dispatch-graph.h"
#include "pipe-prefetch-hints.h"

#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-impl.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
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

// Guards phantom-row detection (note_batch_tokens below): only meaningful
// when the verify/draft batch is actually padded with phantom mask tokens,
// which since the 2026-08-24 split is WP_SPEC_CONST_WIDTH's decision alone
// -- WP_DS4_CONST_SHAPE=1 no longer implies padding (see
// tools/server/server-context.cpp server_spec_const_width()).
bool const_shape_enabled() {
    static const bool enabled = [] {
        const char * width = std::getenv("WP_SPEC_CONST_WIDTH");
        return width != nullptr && std::atoi(width) > 0;
    }();
    return enabled;
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
    int32_t            chunk_index = 0;
    int32_t            chunk_count = 1;
    int64_t            token_offset = 0;
    // hparams.swiglu_clamp_exp[layer]; <= 0 means no clamp. See the note on
    // pipe_expert_dispatch_req::swiglu_clamp -- the spine's own clamped SwiGLU
    // is unreachable on the dispatch path, so the worker must apply it and the
    // limit has to travel with every request.
    float              swiglu_clamp = 0.0f;
    ggml_tensor *      issued = nullptr;
    ggml_tensor *      full_activations = nullptr;
    ggml_tensor *      full_selected_experts = nullptr;
    ggml_tensor *      full_weights = nullptr;
    dispatcher::dispatch_handle handle = 0;
    uint64_t           seq_id = 0;
    dispatch_stats     stats;
};

graph_dispatcher::graph_dispatcher(const std::string & endpoints,
                                   int32_t             n_embd,
                                   int32_t             n_ff_exp,
                                   int32_t             n_expert,
                                   int32_t             n_expert_used,
                                   int32_t             last_no_defer_layer,
                                   int32_t             phantom_token) :
    remote(parse_endpoints(endpoints)),
    phantom_token_(phantom_token),
    collect_stats_(dispatch_stats_enabled()) {
    static std::once_flag chunks_log_once;
    std::call_once(chunks_log_once, [this] {
        LLAMA_LOG_WARN("expert dispatch: wp dispatch chunks=%d\n", remote.dispatch_chunks());
    });
    if (layer_trace_enabled()) {
        const char * p = std::getenv("WP_DS4_LAYER_TRACE");
        layer_trace_ = std::fopen(p, "w");
        if (layer_trace_ != nullptr) {
            LLAMA_LOG_WARN("expert dispatch: WP_DS4_LAYER_TRACE=%s (one line per layer)\n", p);
        }
    }
    if (const char * p = std::getenv("WP_DISPATCH_NULL"); p != nullptr && p[0] == '1') {
        LLAMA_LOG_WARN(
            "expert dispatch: WP_DISPATCH_NULL=1 (TIMING PROBE: routed experts "
            "zeroed, workers never contacted, outputs are garbage)\n");
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
        while (remote.has_open_dispatch()) {
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

void graph_dispatcher::layer_trace_issue_return(int32_t layer, dispatch_clock::time_point time) noexcept {
    if (layer_trace_ == nullptr) {
        return;
    }
    layer_trace_record & record = layer_traces_[layer];
    record.dense_ns = 0;
    record.dense_started = time;
    record.dense_active = true;
}

void graph_dispatcher::layer_trace_wait_entry(int32_t layer, dispatch_clock::time_point time) noexcept {
    if (layer_trace_ == nullptr) {
        return;
    }
    const auto it = layer_traces_.find(layer);
    if (it == layer_traces_.end() || !it->second.dense_active) {
        return;
    }
    it->second.dense_ns += elapsed_ns(it->second.dense_started, time);
    it->second.dense_active = false;
}

void graph_dispatcher::spine_profile_begin(dispatch_clock::time_point begin) noexcept {
    spine_profile_ = {};
    spine_profile_.begin = begin;
    spine_profile_.active = true;
}

graph_dispatcher::spine_profile_stats graph_dispatcher::spine_profile_end(dispatch_clock::time_point end) noexcept {
    spine_profile_stats stats;
    if (!spine_profile_.active) {
        return stats;
    }
    stats = spine_profile_.stats;
    if (spine_profile_.have_first_issue) {
        stats.ns_before_first_issue = elapsed_ns(spine_profile_.begin, spine_profile_.first_issue);
    }
    if (spine_profile_.have_last_wait) {
        stats.ns_after_last_wait = elapsed_ns(spine_profile_.last_wait_end, end);
    }
    const uint64_t accounted = stats.ns_dispatch_issue_total + stats.ns_dispatch_wait_total +
        stats.ns_between_issue_and_wait_total + stats.ns_before_first_issue + stats.ns_after_last_wait;
    const uint64_t ns_graph = elapsed_ns(spine_profile_.begin, end);
    stats.ns_gaps = ns_graph > accounted ? ns_graph - accounted : 0;
    spine_profile_.active = false;
    return stats;
}

void graph_dispatcher::spine_profile_issue_begin(dispatch_clock::time_point time) noexcept {
    if (!spine_profile_.active) {
        return;
    }
    spine_profile_.issue_started = time;
    spine_profile_.issue_active = true;
    if (!spine_profile_.have_first_issue) {
        spine_profile_.first_issue = time;
        spine_profile_.have_first_issue = true;
    }
}

void graph_dispatcher::spine_profile_issue_end(int32_t layer, dispatch_clock::time_point time) noexcept {
    if (!spine_profile_.active || !spine_profile_.issue_active) {
        return;
    }
    spine_profile_.stats.ns_dispatch_issue_total += elapsed_ns(spine_profile_.issue_started, time);
    spine_profile_.issue_ended[layer] = time;
    ++spine_profile_.stats.n_layers;
    spine_profile_.issue_active = false;
}

void graph_dispatcher::spine_profile_wait_begin(int32_t layer, dispatch_clock::time_point time) noexcept {
    if (!spine_profile_.active) {
        return;
    }
    const auto it = spine_profile_.issue_ended.find(layer);
    if (it != spine_profile_.issue_ended.end()) {
        spine_profile_.stats.ns_between_issue_and_wait_total += elapsed_ns(it->second, time);
    }
    spine_profile_.issue_started = time;
    spine_profile_.issue_active = true;
}

void graph_dispatcher::spine_profile_wait_end(dispatch_clock::time_point time) noexcept {
    if (!spine_profile_.active || !spine_profile_.issue_active) {
        return;
    }
    spine_profile_.stats.ns_dispatch_wait_total += elapsed_ns(spine_profile_.issue_started, time);
    spine_profile_.last_wait_end = time;
    spine_profile_.have_last_wait = true;
    spine_profile_.issue_active = false;
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
    const auto first = op_contexts.find(layer * 2);
    const auto second = op_contexts.find(layer * 2 + 1);
    const bool chunked = second != op_contexts.end() && second->second != nullptr &&
                         second->second->chunk_count > 1;
    std::string labels = "(" + std::to_string(layer) + ",0," +
                         std::to_string(first != op_contexts.end() && first->second != nullptr
                                            ? first->second->seq_id : 0) + ")";
    if (chunked) {
        labels += " (" + std::to_string(layer) + ",1," +
                  std::to_string(second->second->seq_id) + ")";
    }
    std::fprintf(layer_trace_, "DS4 layer=%d chunks=%d labels=%s dense_ns=%llu encode_ns=%llu send_ns=%llu recv_ns=%llu decode_ns=%llu scatter_ns=%llu\n",
                 layer,
                 chunked ? 2 : 1,
                 labels.c_str(),
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
                                      float          swiglu_clamp,
                                      bool            chunked) {
    if (layer < 0) {
        throw std::invalid_argument("expert dispatch requires a non-negative layer");
    }
    if (activations->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32 ||
        weights->type != GGML_TYPE_F32) {
        throw std::invalid_argument("expert dispatch requires F32 activations, I32 expert ids, and F32 weights");
    }

    if (chunked) {
        const int64_t n_tokens = activations->ne[1];
        const int64_t n_first = n_tokens / 2;
        const int64_t n_second = n_tokens - n_first;
        const int64_t n_expert_used = selected_experts->ne[0];
        ggml_tensor * activations_a = ggml_view_2d(ctx, activations, activations->ne[0], n_first,
                                                   activations->nb[1], 0);
        ggml_tensor * activations_b = ggml_view_2d(ctx, activations, activations->ne[0], n_second,
                                                   activations->nb[1], n_first * activations->nb[1]);
        ggml_tensor * selected_a = ggml_view_2d(ctx, selected_experts, n_expert_used, n_first,
                                                selected_experts->nb[1], 0);
        ggml_tensor * selected_b = ggml_view_2d(ctx, selected_experts, n_expert_used, n_second,
                                                selected_experts->nb[1], n_first * selected_experts->nb[1]);
        ggml_tensor * weights_a = ggml_view_3d(ctx, weights, weights->ne[0], weights->ne[1], n_first,
                                               weights->nb[1], weights->nb[2], 0);
        ggml_tensor * weights_b = ggml_view_3d(ctx, weights, weights->ne[0], weights->ne[1], n_second,
                                               weights->nb[1], weights->nb[2], n_first * weights->nb[2]);
        ggml_tensor * issue_a = build_issue(ctx, activations_a, selected_a, weights_a, layer, swiglu_clamp,
                                            0, 2, nullptr, activations, selected_experts, weights, 0);
        ggml_tensor * issue_b = build_issue(ctx, activations_b, selected_b, weights_b, layer, swiglu_clamp,
                                            1, 2, issue_a, activations, selected_experts, weights, n_first);
        GGML_UNUSED(issue_b);
        return build_wait(ctx, layer);
    }

    auto & context = op_contexts[layer * 2];
    if (!context) {
        context.reset(new op_context{ this, layer });
    }
    context->chunk_index = 0;
    context->chunk_count = 1;
    context->token_offset = 0;
    context->full_activations = nullptr;
    context->full_selected_experts = nullptr;
    context->full_weights = nullptr;
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
                                            float          swiglu_clamp,
                                            int32_t        chunk_index,
                                            int32_t        chunk_count,
                                            ggml_tensor *  issue_dependency,
                                            ggml_tensor *  full_activations,
                                            ggml_tensor *  full_selected_experts,
                                            ggml_tensor *  full_weights,
                                            int64_t        token_offset) {
    if (layer < 0) {
        throw std::invalid_argument("expert dispatch requires a non-negative layer");
    }
    if (activations->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32 ||
        weights->type != GGML_TYPE_F32) {
        throw std::invalid_argument("expert dispatch requires F32 activations, I32 expert ids, and F32 weights");
    }

    if (chunk_count < 1 || chunk_index < 0 || chunk_index >= chunk_count) {
        throw std::invalid_argument("expert dispatch has an invalid chunk index");
    }
    if (issue_dependency != nullptr) {
        ggml_tensor * gate = ggml_scale(ctx, ggml_view_1d(ctx, issue_dependency, 1, 0), 0.0f);
        weights = ggml_add(ctx, weights, gate);
    }
    const int32_t key = layer * 2 + chunk_index;
    auto & context = op_contexts[key];
    if (!context) {
        context.reset(new op_context{ this, layer });
    }
    context->chunk_index         = chunk_index;
    context->chunk_count         = chunk_count;
    context->token_offset        = token_offset;
    context->swiglu_clamp = swiglu_clamp;
    context->full_activations    = full_activations;
    context->full_selected_experts = full_selected_experts;
    context->full_weights        = full_weights;
    ggml_tensor * issued =
        ggml_map_custom3(ctx, activations, selected_experts, weights, compute_issue, 1, context.get());
    context->issued = issued;
    return issued;
}

ggml_tensor * graph_dispatcher::after_issue(ggml_context * ctx, ggml_tensor * tensor, int32_t layer) {
    if (tensor == nullptr) {
        throw std::invalid_argument("after_issue requires a tensor");
    }
    const auto it = op_contexts.find(layer * 2);
    if (it == op_contexts.end() || it->second == nullptr || it->second->issued == nullptr) {
        throw std::runtime_error("after_issue has no issue node for layer " + std::to_string(layer));
    }
    // ACC keeps both src edges, so the FFN that consumes this result cannot
    // start before issue. Inplace: one element, no residual copy. `tensor` is
    // the shexp *input* (a scale/view of the residual), not the issue
    // activations themselves — issue has already copied those to the CPU.
    ggml_tensor * gate = ggml_view_1d(ctx, it->second->issued, 1, 0);
    if (it->second->chunk_count > 1) {
        const auto next = op_contexts.find(layer * 2 + 1);
        if (next == op_contexts.end() || next->second == nullptr || next->second->issued == nullptr) {
            throw std::runtime_error("after_issue has incomplete chunk issues for layer " + std::to_string(layer));
        }
        gate = ggml_add(ctx, gate, ggml_view_1d(ctx, next->second->issued, 1, 0));
    }
    ggml_tensor * zero = ggml_scale(ctx, gate, 0.0f);
    return ggml_acc_inplace(ctx, tensor, zero, tensor->nb[1], tensor->nb[2], tensor->nb[3], 0);
}

ggml_tensor * graph_dispatcher::build_wait(ggml_context * ctx, int32_t layer) {
    const auto it = op_contexts.find(layer * 2);
    if (it == op_contexts.end() || it->second == nullptr || it->second->issued == nullptr) {
        throw std::runtime_error("build_wait has no issue node for layer " + std::to_string(layer));
    }
    // Two copies of `issued`: compute_wait's signature is still custom2, but
    // both srcs are the CPU issue node. A GPU shexp src here made wait a
    // GPU→CPU split input and serialized recv behind shexp.
    ggml_tensor * issued = it->second->issued;
    if (it->second->chunk_count == 1) {
        return ggml_map_custom2(ctx, issued, issued, compute_wait, 1, it->second.get());
    }
    const auto next = op_contexts.find(layer * 2 + 1);
    if (next == op_contexts.end() || next->second == nullptr || next->second->issued == nullptr) {
        throw std::runtime_error("build_wait has incomplete chunk issues for layer " + std::to_string(layer));
    }
    ggml_tensor * wait_a = ggml_map_custom2(ctx, issued, next->second->issued,
                                            compute_wait, 1, it->second.get());
    ggml_tensor * wait_b = ggml_map_custom2(ctx, next->second->issued, wait_a,
                                            compute_wait, 1, next->second.get());
    return ggml_concat(ctx, wait_a, wait_b, 1);
}

void graph_dispatcher::begin_graph_build(ggml_context * ctx) {
    graph_build_ctx_ = ctx;
    last_chunked_issue_layer_ = -1;
    last_expanded_wait_layer_ = -1;
}

bool graph_dispatcher::begin_chunked_issue_build(ggml_context * ctx, int32_t layer) {
    if (graph_build_ctx_ != ctx) {
        begin_graph_build(ctx);
    }

    const int32_t previous_layer = layer - 1;
    const bool ordered = last_chunked_issue_layer_ != previous_layer ||
                         last_expanded_wait_layer_ >= previous_layer;
    if (!ordered) {
        std::fprintf(stderr,
                     "expert dispatch graph order: layer %d issue created before layer %d wait expanded\n",
                     layer, previous_layer);
    }
    last_chunked_issue_layer_ = layer;
    return ordered;
}

void graph_dispatcher::note_wait_expanded(ggml_context * ctx, int32_t layer) {
    if (graph_build_ctx_ != ctx) {
        begin_graph_build(ctx);
    }
    last_expanded_wait_layer_ = std::max(last_expanded_wait_layer_, layer);
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

bool graph_dispatcher::reuse_last_enabled() {
    // Not cached: the unit test has to A/B the flag in one process, and a
    // getenv per hint is lost in the noise next to a 9 MiB page-in.
    const char * value = std::getenv("WP_HINT_REUSE_LAST");
    return value != nullptr && value[0] == '1';
}

// Decode/spec-verify only. Same 32 that WP_DEFER_MAX_WIDTH / spec prefill
// gate / LAST_K use to tell a verify batch from a 2048-token prefill.
static uint32_t reuse_max_tokens() {
    static const uint32_t value = [] {
        const char * e = std::getenv("WP_HINT_REUSE_MAX_TOKENS");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 32;
        return v > 0 ? (uint32_t) v : 32u;
    }();
    return value;
}

// WP_HINT_REUSE_PAGES: max expert-pages a reuse flush may offer. Default 32
// so a 64-deep worker queue still has room for hash CERTAIN + router2.
// 0 = uncapped (the 2026-08-19 flood: 43 layers x ~6 ids into a 64-page queue).
// Not cached: the unit test A/Bs the budget in one process.
static size_t reuse_page_budget() {
    const char * e = std::getenv("WP_HINT_REUSE_PAGES");
    if (e == nullptr || e[0] == '\0') {
        return 32;
    }
    const long v = strtol(e, nullptr, 10);
    return v > 0 ? (size_t) v : 0;
}

void graph_dispatcher::note_dispatched_experts(
        int32_t layer, const std::vector<pipe_expert_assignment> & assignments,
        uint32_t n_tokens) noexcept {
    try {
        if (!reuse_last_enabled() || !hint_enabled() || layer < 0 ||
            n_tokens == 0 || n_tokens > reuse_max_tokens() || assignments.empty()) {
            return;
        }
        // Rank by how many tokens in this ubatch actually used the expert, so a
        // verify-batch union that exceeds PREFETCH_HINT_MAX_EXPERTS keeps the
        // ones shared across positions, not the lowest ids.
        struct ranked {
            int32_t expert_id;
            int     n_used;
        };
        std::vector<ranked> ranked_experts;
        ranked_experts.reserve(assignments.size());
        for (const pipe_expert_assignment & assignment : assignments) {
            if (assignment.expert_id < 0) {
                continue;
            }
            int n_used = 0;
            for (float w : assignment.weights) {
                if (w != 0.0f) {
                    ++n_used;
                }
            }
            if (n_used > 0) {
                ranked_experts.push_back({ assignment.expert_id, n_used });
            }
        }
        if (ranked_experts.empty()) {
            return;
        }
        const size_t cap = (size_t) PREFETCH_HINT_MAX_EXPERTS;
        if (ranked_experts.size() > cap) {
            std::nth_element(ranked_experts.begin(),
                             ranked_experts.begin() + (ptrdiff_t) cap,
                             ranked_experts.end(),
                             [](const ranked & a, const ranked & b) {
                                 return a.n_used != b.n_used ? a.n_used > b.n_used
                                                             : a.expert_id < b.expert_id;
                             });
            ranked_experts.resize(cap);
        }
        reuse_set set;
        set.n_tokens = n_tokens;
        set.experts.reserve(ranked_experts.size());
        for (const ranked & e : ranked_experts) {
            set.experts.push_back(e.expert_id);
        }
        std::sort(set.experts.begin(), set.experts.end());
        last_dispatched_[layer] = std::move(set);
    } catch (...) {
        // Advisory. A dropped reuse set costs one missed hint.
    }
}

size_t graph_dispatcher::flush_reuse_hints() noexcept {
    // MAD-LAB DS4-Flash pipeline-streams: see io_mutex_'s declaration.
    std::lock_guard<std::recursive_mutex> io_lock(io_mutex_);
    if (!reuse_last_enabled() || !hint_enabled() || last_dispatched_.empty()) {
        return 0;
    }
    size_t sent = 0;
    try {
        const size_t                 budget = reuse_page_budget();
        size_t                       pages  = 0;
        const std::vector<int32_t> & hash_layers = oracle_.layers();
        // Map is layer-ascending. Skip hash layers (tid2eid already exact),
        // emit soonest first, stop when the next FULL layer would exceed the
        // page budget. A partial layer is not offered: n_expert_used or nothing.
        for (const auto & entry : last_dispatched_) {
            if (std::binary_search(hash_layers.begin(), hash_layers.end(), entry.first)) {
                continue;
            }
            const std::vector<int32_t> & experts = entry.second.experts;
            if (experts.empty()) {
                continue;
            }
            if (budget != 0 && pages + experts.size() > budget) {
                break;
            }
            std::vector<int32_t> & previous = last_reuse_hint_[entry.first];
            if (previous == experts) {
                pages += experts.size();
                continue;
            }
            const size_t n = remote.send_prefetch_hints(entry.first, experts,
                                                        PIPE_HINT_PREDICTED,
                                                        entry.second.n_tokens);
            if (n == 0) {
                // in_flight / no route: do not mark sent, retry this layer next
                // flush. Breaking keeps soonest-first instead of skipping to L40.
                break;
            }
            previous = experts;
            pages += experts.size();
            sent += n;
        }
    } catch (...) {
    }
    return sent;
}

void graph_dispatcher::clear_reuse_hints() noexcept {
    last_dispatched_.clear();
    last_reuse_hint_.clear();
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
    // MAD-LAB DS4-Flash pipeline-streams: see io_mutex_'s declaration. This
    // is the exact function whose doc comment already said "MUST NOT be
    // called with a dispatch in flight: it writes to the same sockets" --
    // now enforced rather than just documented.
    std::lock_guard<std::recursive_mutex> io_lock(io_mutex_);
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

size_t graph_dispatcher::pred_queue_depth() {
    static const size_t value = [] {
        const char * v = std::getenv("WP_HINT_QUEUE_DEPTH");
        if (v == nullptr || v[0] == '\0') {
            return (size_t) 0;   // legacy one-slot mailbox, byte-identical
        }
        const long n = strtol(v, nullptr, 10);
        return n > 0 ? (size_t) n : (size_t) 0;
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
        const size_t depth = pred_queue_depth();
        {
            std::lock_guard<std::mutex> lock(pred_mutex_);
            ++pred_offered_;
            // First snapshot this decode is the one with lead. Overwriting it
            // with a later layer (latest-wins) is what made K=7 score middle
            // layers with no lead. Drop the NEW job instead. This gate decides
            // WHAT gets scored (the earliest layer, for maximum lead) and is
            // unchanged by WP_HINT_QUEUE_DEPTH -- the queue only changes
            // whether an ACCEPTED snapshot can survive the predictor still
            // being busy with a previous one.
            if (pred_snapshot_taken_) {
                ++pred_dropped_;
                return;
            }
            pred_snapshot_taken_ = true;

            if (depth == 0) {
                // Legacy path: one slot, unconditional overwrite, byte-identical
                // to the mailbox this replaces when the knob is unset. If the
                // predictor has not yet drained a previous decode's snapshot
                // (pred_inbox_.valid still true), this silently replaces it --
                // exactly as before. That silent loss is invisible to
                // pred_dropped_/pred_offered_/pred_scored_, which is the gap
                // WP_HINT_QUEUE_DEPTH exists to close.
                pred_inbox_.layer    = layer;
                pred_inbox_.n_tokens = n_tokens;
                pred_inbox_.activations.assign(activations.begin(), activations.end());
                pred_inbox_.valid    = true;
            } else {
                // Bounded FIFO: never overwrites a queued-but-undrained
                // snapshot. Full is the only way to lose one here, and that
                // loss is counted (pred_queue_overflow_) and distinct from
                // pred_dropped_ (same-decode duplicate suppression above).
                if (pred_queue_.size() >= depth) {
                    ++pred_queue_overflow_;
                    return;
                }
                pred_job job;
                job.layer    = layer;
                job.n_tokens = n_tokens;
                job.activations.assign(activations.begin(), activations.end());
                job.valid    = true;
                pred_queue_.push_back(std::move(job));
                pred_queue_hwm_ = std::max(pred_queue_hwm_, pred_queue_.size());
            }
        }
        pred_cv_.notify_one();
    } catch (...) {
        // Swallow. A dropped snapshot costs one hint, never the decode.
    }
}

void graph_dispatcher::predictor_loop() {
    // Reused across every job this thread ever scores (single-threaded
    // consumer, so no locking needed): see router2_scratch on why this
    // matters once WP_HINT_QUEUE_DEPTH lets more than one snapshot per
    // decode reach the K-deep GEMV loop below.
    router2_scratch scratch;
    for (;;) {
        pred_job job;
        {
            std::unique_lock<std::mutex> lock(pred_mutex_);
            pred_cv_.wait(lock, [this] {
                return pred_stop_ || pred_inbox_.valid || !pred_queue_.empty();
            });
            if (pred_stop_) {
                return;
            }
            if (!pred_queue_.empty()) {
                job = std::move(pred_queue_.front());
                pred_queue_.pop_front();
            } else {
                job = std::move(pred_inbox_);
                pred_inbox_.valid = false;
            }
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
                    n_expert, n_embd, m, conf, scratch);
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
    // MAD-LAB DS4-Flash pipeline-streams: see io_mutex_'s declaration.
    // Recursive because this is called from INSIDE compute()/compute_issue()
    // (same thread, already holding io_mutex_) as well as, in principle,
    // standalone -- a plain mutex would self-deadlock on the nested case.
    std::lock_guard<std::recursive_mutex> io_lock(io_mutex_);
    try {
        std::map<int32_t, pred_result> ready;
        {
            std::lock_guard<std::mutex> lock(pred_mutex_);
            if (pred_ready_.empty()) {
                return;
            }
            ready.swap(pred_ready_);
        }
        // Cap so reuse (32) + router2 fits the 64-deep worker queue. Default 16
        // PAGES PER DECODE, not per flush. 0 = uncapped. Soonest layer first,
        // all-or-nothing per layer.
        static const size_t page_budget = [] {
            const char * e = std::getenv("WP_HINT_ROUTER2_PAGES");
            if (e == nullptr || e[0] == '\0') {
                return (size_t) 16;
            }
            const long v = strtol(e, nullptr, 10);
            return v > 0 ? (size_t) v : (size_t) 0;
        }();
        for (auto & entry : ready) {
            const std::vector<int32_t> & experts = entry.second.experts;
            if (experts.empty()) {
                continue;
            }
            if (page_budget != 0 &&
                router2_pages_this_decode_ + experts.size() > page_budget) {
                break;
            }
            std::vector<int32_t> & previous = last_pred_hint_[entry.first];
            if (previous == experts) {
                continue;
            }
            const size_t n = remote.send_prefetch_hints(entry.first, experts,
                                                        PIPE_HINT_PREDICTED,
                                                        entry.second.n_tokens);
            if (n == 0) {
                break;
            }
            previous = experts;
            router2_pages_this_decode_ += experts.size();
        }
    } catch (...) {
        // Swallow. A broken socket surfaces on the next real dispatch.
    }
}

void graph_dispatcher::prefetch_layer_ahead(int32_t layer, uint32_t n_tokens) noexcept {
    // MAD-LAB DS4-Flash pipeline-streams: see io_mutex_'s declaration.
    std::lock_guard<std::recursive_mutex> io_lock(io_mutex_);
    try {
        static const bool enabled = [] {
            const char * v = std::getenv("WP_PREFILL_LAYER_AHEAD");
            return v != nullptr && v[0] == '1';
        }();
        static const uint32_t width = [] {
            const char * e = std::getenv("WP_PREFILL_LAYER_AHEAD_WIDTH");
            const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 8;
            return v > 0 ? (uint32_t) v : 8u;
        }();
        if (!enabled || n_tokens <= width) {
            return;
        }
        int32_t nxt = -1;
        for (const worker_info & worker : remote.workers()) {
            const auto it = std::upper_bound(
                worker.layers.begin(), worker.layers.end(), layer);
            if (it != worker.layers.end() && (nxt < 0 || *it < nxt)) {
                nxt = *it;
            }
        }
        if (nxt < 0) {
            return;
        }
        const int32_t n_expert = remote.n_expert();
        if (n_expert <= 0) {
            return;
        }
        // One frame of the full expert set; send_prefetch_hints partitions by
        // worker. Protocol allows more than PREFETCH_HINT_MAX_EXPERTS (that
        // cap is generation-side for decode). A whole-slice frame is how the
        // worker tells this apart from a decode CERTAIN hint and keeps it off
        // the 64-deep spec queue.
        std::vector<int32_t> experts((size_t) n_expert);
        for (int32_t e = 0; e < n_expert; ++e) {
            experts[(size_t) e] = e;
        }
        (void) remote.send_prefetch_hints(nxt, experts, PIPE_HINT_CERTAIN, n_tokens);
    } catch (...) {
    }
}

size_t graph_dispatcher::prefetch_ngram_for_tokens(const int32_t * tokens, size_t n_tokens) noexcept {
    // MAD-LAB DS4-Flash pipeline-streams: see io_mutex_'s declaration.
    std::lock_guard<std::recursive_mutex> io_lock(io_mutex_);
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
            phantom_rows_.clear();
            return;
        }
        phantom_rows_.clear();
        if (const_shape_enabled() && phantom_token_ >= 0) {
            for (size_t i = 0; i < n_tokens; ++i) {
                if (tokens[i] == phantom_token_) {
                    phantom_rows_.assign(n_tokens, 0);
                    for (size_t j = i; j < n_tokens; ++j) {
                        phantom_rows_[j] = tokens[j] == phantom_token_;
                    }
                    break;
                }
            }
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

bool graph_dispatcher::is_phantom_row(int64_t token) const noexcept {
    return token >= 0 && (size_t) token < phantom_rows_.size() && phantom_rows_[(size_t) token] != 0;
}

void graph_dispatcher::zero_phantom_rows(std::vector<float> & result, int64_t n_tokens, int64_t n_embd,
                                         int64_t token_offset) const noexcept {
    for (int64_t token = 0; token < n_tokens; ++token) {
        if (is_phantom_row(token + token_offset)) {
            std::fill(result.begin() + token*n_embd, result.begin() + (token + 1)*n_embd, 0.0f);
        }
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
    // Hold the dispatcher lock across the whole llama_decode() call. A draft
    // context can borrow this dispatcher, so per-callback locking still allows
    // it to issue while the target's chunk handles are open.
    io_mutex_.lock();
    phantom_rows_.clear();
    router2_pages_this_decode_ = 0;
    pred_snapshot_taken_       = false;
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
    decode_ns_fold_overlapped_            = 0;
    decode_n_partials_folded_early_       = 0;
    decode_first_await_in_flight_        = 0;
    decode_max_in_flight_                = 0;
    decode_n_tokens_                     = 0;
    decode_workers_.clear();
    for (const worker_info & worker : remote.workers()) {
        decode_workers_.push_back({ worker.endpoint });
    }
}

void graph_dispatcher::end_decode() noexcept {
    // begin_decode() holds this recursive lock for the decode scope.
    std::unique_lock<std::recursive_mutex> io_lock(io_mutex_, std::adopt_lock);
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
            //
            // queue=<depth> (0 = legacy one-slot mailbox) overflow=<count> is
            // WP_HINT_QUEUE_DEPTH's own loss counter -- distinct from `dropped`
            // above, which only counts same-decode duplicate snapshots (the
            // first-wins gate, unaffected by the queue). overflow is the count
            // that should read ~0 once the depth covers the real
            // predictor-vs-offer-rate lag; hwm is how deep the backlog got, so
            // hwm pinned at queue tells you the depth itself is still too
            // shallow (or the K-deep GEMV loop is the actual bottleneck and no
            // depth will fix it -- see router2_scratch).
            std::fprintf(stderr,
                "expert dispatch predictor: offered=%llu scored=%llu dropped=%llu (%.1f%% dropped) "
                "queue=%zu overflow=%llu hwm=%zu\n",
                (unsigned long long) offered,
                (unsigned long long) pred_scored(),
                (unsigned long long) dropped,
                100.0 * (double) dropped / (double) offered,
                pred_queue_depth(),
                (unsigned long long) pred_queue_overflow(),
                pred_queue_high_water());
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
            "sent_in_flight=%llu send_failed=%llu no_route=%llu "
            "skip_dynamic=%llu skip_in_flight=%llu\n",
            oracle_.layers().size(),
            (unsigned long long) hstats.n_frames,
            (unsigned long long) hstats.n_experts,
            (unsigned long long) hstats.n_sent_in_flight,
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
        "unpack=%.2f ms total=%.2f ms (%.3f ms/tok) ns_fold_overlapped=%.2f ms "
        "n_partials_folded_early=%llu first_await_in_flight avg=%.1f max_in_flight=%zu "
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
        decode_ns_fold_overlapped_ * ns_to_ms,
        (unsigned long long) decode_n_partials_folded_early_,
        (double) decode_first_await_in_flight_ / (double) decode_layers_,
        decode_max_in_flight_,
        n_workers());
    // WP_UNPACK_OVERLAP=1 (see pipe-expert-dispatcher.cpp's
    // unpack_overlap_enabled() for the mechanism): the dispatcher now decodes
    // each worker's response as its socket becomes ready instead of strictly
    // after the previous fixed-order worker, so a decode that used to happen
    // entirely after the last recv (and show up in `unpack` above) can now
    // happen while an earlier-in-fixed-order-but-slower worker is still on
    // the wire. That moves its cost into `wait` above instead. The SUM
    // (issue+wait+unpack == total) is unaffected; only the split between wait
    // and unpack shifts. Printed once per decode block, same cadence as the
    // line above, so a run's log always says which attribution is in effect.
    if (std::getenv("WP_UNPACK_OVERLAP") != nullptr && std::getenv("WP_UNPACK_OVERLAP")[0] == '1') {
        LLAMA_LOG_WARN(
            "expert dispatch: WP_UNPACK_OVERLAP=1 -- wait/unpack split above may attribute "
            "some decode cost to `wait` that previously showed as `unpack` (see "
            "unpack_overlap_enabled() in pipe-expert-dispatcher.cpp); total is unaffected\n");
    }
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
        // MAD-LAB DS4-Flash pipeline-streams: per-dispatcher socket
        // exclusivity -- see io_mutex_'s declaration for why this is needed
        // even though a single graph_dispatcher used to have exactly one
        // caller thread. Recursive: flush_predicted_hints() below re-enters
        // this same lock on the same thread.
        std::lock_guard<std::recursive_mutex> io_lock(owner->io_mutex_);
        if (owner->failed()) {
            ggml_set_zero(dst);
            return;
        }
        // WP_DISPATCH_NULL=1 -- TIMING PROBE ONLY (outputs are garbage; never
        // pair with a quality gate). Zero the routed-expert result and return
        // without contacting any worker: the decode then costs only the
        // spine's own compute + graph scaffolding. Hop-theory probe A,
        // 2026-08-22. Combined op only (run with WP_DISPATCH_SPLIT_SHEXP=0);
        // decode_layers_ still counts so WP_FORWARD_LOG emits.
        static const bool dispatch_null = [] {
            const char * e = std::getenv("WP_DISPATCH_NULL");
            return e != nullptr && e[0] == '1';
        }();
        if (dispatch_null) {
            ggml_set_zero(dst);
            if (owner->decode_active_) {
                ++owner->decode_layers_;
            }
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
            if (owner->is_phantom_row(token)) {
                continue;
            }
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
        owner->note_dispatched_experts(context->layer, assignments, (uint32_t) n_tokens);
        if (const char * capture_prefix = std::getenv("WP_PREDICT_CAPTURE");
            capture_prefix != nullptr && capture_prefix[0] != '\0') {
            owner->capture_routing(capture_prefix, context->layer, wire_activations,
                                   selected_experts, n_tokens, n_expert_used);
        }
        owner->prefetch_layer_ahead(context->layer, (uint32_t) n_tokens);

        const uint64_t           seq_id = owner->next_seq_id.fetch_add(1, std::memory_order_relaxed);
        // dispatch() issues deferred reads before returning, waits only for
        // immediate experts, and folds the previous layer's deferred partials
        // into the returned block (residual path for layer N+1).
        const dispatcher::dispatch_handle handle = owner->remote.begin_dispatch(
            context->layer, seq_id, (uint32_t) n_tokens, wire_activations, assignments,
            context->swiglu_clamp, 0, nullptr, 0, "graph_dispatcher::compute");
        dispatch_stats layer_stats;
        std::vector<float> result = owner->remote.finish_dispatch(handle, &layer_stats);
        owner->zero_phantom_rows(result, n_tokens, n_embd);
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
            owner->decode_ns_fold_overlapped_ += layer_stats.ns_fold_overlapped;
            owner->decode_n_partials_folded_early_ += layer_stats.n_partials_folded_early;
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
        const bool trace_layer = owner->layer_trace_ != nullptr &&
                                 (context->chunk_count == 1 || context->chunk_index == 0);
        const bool trace_profile = owner->spine_profile_.active &&
                                   (context->chunk_count == 1 || context->chunk_index == 0);
        // MAD-LAB DS4-Flash pipeline-streams: see io_mutex_'s declaration.
        std::lock_guard<std::recursive_mutex> io_lock(owner->io_mutex_);
        if (owner->failed()) {
            if (dst->data != nullptr && dst->type == GGML_TYPE_F32 && ggml_nelements(dst) > 0) {
                *static_cast<float *>(dst->data) = 0.0f;
            }
            return;
        }
        if (context->handle != 0 && owner->remote.has_open_dispatch(context->handle)) {
            if (dst->data != nullptr && dst->type == GGML_TYPE_F32 && ggml_nelements(dst) > 0) {
                *static_cast<float *>(dst->data) = 0.0f;
            }
            return;
        }
        if (trace_profile) {
            owner->spine_profile_issue_begin(dispatch_clock::now());
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

        const auto make_assignments = [&](const ggml_tensor * selected,
                                          const ggml_tensor * route_weights,
                                          int64_t             route_tokens,
                                          int64_t             token_offset) {
            std::map<int32_t, pipe_expert_assignment> by_expert;
            for (int64_t token = 0; token < route_tokens; ++token) {
                if (owner->is_phantom_row(token + token_offset)) {
                    continue;
                }
                std::set<int32_t> token_experts;
                for (int64_t slot = 0; slot < n_expert_used; ++slot) {
                    const int     index  = (int) (token * n_expert_used + slot);
                    const int32_t expert = ggml_get_i32_1d(selected, index);
                    if (!token_experts.insert(expert).second) {
                        throw std::runtime_error("expert dispatch received a repeated expert for one token");
                    }
                    auto inserted = by_expert.emplace(expert, pipe_expert_assignment{});
                    if (inserted.second) {
                        inserted.first->second.expert_id = expert;
                        inserted.first->second.weights.resize((size_t) route_tokens, 0.0f);
                    }
                    inserted.first->second.weights[(size_t) token] = ggml_get_f32_1d(route_weights, index);
                }
            }
            std::vector<pipe_expert_assignment> result;
            result.reserve(by_expert.size());
            for (auto & entry : by_expert) {
                result.push_back(std::move(entry.second));
            }
            return result;
        };

        std::vector<pipe_expert_assignment> assignments =
            make_assignments(selected_experts, weights, n_tokens, context->token_offset);

        std::vector<pipe_expert_assignment> full_assignments;
        const std::vector<pipe_expert_assignment> * layer_assignments = nullptr;
        uint32_t layer_n_tokens = 0;
        if (context->chunk_index == 0) {
            const ggml_tensor * full_activations = context->full_activations != nullptr
                ? context->full_activations : activations;
            const ggml_tensor * full_selected = context->full_selected_experts != nullptr
                ? context->full_selected_experts : selected_experts;
            const ggml_tensor * full_weights = context->full_weights != nullptr
                ? context->full_weights : weights;
            const int64_t full_tokens = full_activations->ne[1];
            std::vector<float> full_wire_activations((size_t) n_embd * (size_t) full_tokens);
            if (ggml_is_contiguous(full_activations)) {
                std::memcpy(full_wire_activations.data(), full_activations->data,
                            full_wire_activations.size() * sizeof(float));
            } else {
                for (size_t i = 0; i < full_wire_activations.size(); ++i) {
                    full_wire_activations[i] = ggml_get_f32_1d(full_activations, (int) i);
                }
            }
            full_assignments = make_assignments(full_selected, full_weights, full_tokens, 0);
            layer_assignments = &full_assignments;
            layer_n_tokens = (uint32_t) full_tokens;
            if (owner->router2_topm() > 0) {
                owner->flush_predicted_hints();
                owner->enqueue_prediction(context->layer, full_wire_activations, full_tokens);
            }
            owner->note_dispatched_experts(context->layer, full_assignments, (uint32_t) full_tokens);
            if (const char * capture_prefix = std::getenv("WP_PREDICT_CAPTURE");
                capture_prefix != nullptr && capture_prefix[0] != '\0') {
                owner->capture_routing(capture_prefix, context->layer, full_wire_activations,
                                       full_selected, full_tokens, n_expert_used);
            }
            owner->prefetch_layer_ahead(context->layer, (uint32_t) full_tokens);
        }

        const uint64_t seq_id = owner->next_seq_id.fetch_add(1, std::memory_order_relaxed);
        context->seq_id = seq_id;
        context->handle = owner->remote.begin_dispatch(context->layer, seq_id, (uint32_t) n_tokens,
                                                       wire_activations, assignments, context->swiglu_clamp,
                                                       (uint32_t) context->chunk_index, layer_assignments,
                                                       layer_n_tokens, "graph_dispatcher::compute_issue");
        if (context->chunk_count > 1 && context->chunk_index == 1) {
            static const bool trace = [] {
                const char * e = std::getenv("WP_DISPATCH_CHUNKS_TRACE");
                return e != nullptr && e[0] == '1';
            }();
            static const int trace_layers = [] {
                const char * e = std::getenv("WP_DISPATCH_CHUNKS_TRACE_LAYERS");
                return e != nullptr && e[0] != '\0' ? std::max(1, std::atoi(e)) : 4;
            }();
            const auto first = owner->op_contexts.find(context->layer * 2);
            if (trace && context->layer < trace_layers && first != owner->op_contexts.end() && first->second != nullptr) {
                std::fprintf(stderr,
                             "expert dispatch chunks trace: layer=%d chunk_widths=%lld,%lld "
                             "seq_ids=%llu,%llu order=issue_A issue_B wait_A wait_B\n",
                             context->layer,
                             (long long) first->second->issued->ne[1], (long long) n_tokens,
                             (unsigned long long) first->second->seq_id,
                             (unsigned long long) context->seq_id);
            }
        }
        if (dst->data != nullptr && dst->type == GGML_TYPE_F32 && ggml_nelements(dst) > 0) {
            *static_cast<float *>(dst->data) = 0.0f;
        }
        if (collect_stats) {
            const dispatch_stats layer_stats = owner->remote.stats_for(context->handle);
            owner->decode_n_tokens_ = context->full_activations != nullptr
                ? (uint32_t) context->full_activations->ne[1] : (uint32_t) n_tokens;
            owner->decode_ns_pack_ += elapsed_ns(pack_start, pack_end);
            owner->decode_ns_issue_ += layer_stats.ns_issue;
        }
        if (trace_layer || trace_profile) {
            const auto issue_return = dispatch_clock::now();
            if (trace_layer) {
                owner->layer_trace_issue_return(context->layer, issue_return);
            }
            if (trace_profile) {
                owner->spine_profile_issue_end(context->layer, issue_return);
            }
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
        const bool trace_layer = owner->layer_trace_ != nullptr &&
                                 (context->chunk_count == 1 || context->chunk_index == 0);
        const bool trace_profile = owner->spine_profile_.active &&
                                   (context->chunk_count == 1 || context->chunk_index == 0);
        if (trace_layer || trace_profile) {
            const auto wait_entry = dispatch_clock::now();
            if (trace_layer) {
                owner->layer_trace_wait_entry(context->layer, wait_entry);
            }
            if (trace_profile) {
                owner->spine_profile_wait_begin(context->layer, wait_entry);
            }
        }
        // MAD-LAB DS4-Flash pipeline-streams: see io_mutex_'s declaration.
        std::lock_guard<std::recursive_mutex> io_lock(owner->io_mutex_);
        if (owner->failed()) {
            ggml_set_zero(dst);
            return;
        }
        const bool collect_stats = owner->decode_active_;
        // finish_dispatch() includes the worker wait. Time unpack after it so
        // unpack is the host memcpy into dst, not a second copy of ns_wait.
        dispatch_stats layer_stats;
        std::vector<float> result = owner->remote.finish_dispatch(context->handle, &layer_stats);
        owner->zero_phantom_rows(result, dst->ne[1], owner->remote.n_embd(), context->token_offset);
        context->stats = layer_stats;
        dispatch_stats layer_stats_total = layer_stats;
        if (context->chunk_count > 1 && context->chunk_index == 1) {
            const auto first = owner->op_contexts.find(context->layer * 2);
            if (first != owner->op_contexts.end() && first->second != nullptr) {
                const dispatch_stats & first_stats = first->second->stats;
                layer_stats_total.requests_issued += first_stats.requests_issued;
                layer_stats_total.workers_used += first_stats.workers_used;
                layer_stats_total.ns_pack += first_stats.ns_pack;
                layer_stats_total.ns_issue += first_stats.ns_issue;
                layer_stats_total.ns_wait += first_stats.ns_wait;
                layer_stats_total.ns_unpack += first_stats.ns_unpack;
                layer_stats_total.ns_total += first_stats.ns_total;
                layer_stats_total.ns_fold_overlapped += first_stats.ns_fold_overlapped;
                layer_stats_total.n_partials_folded_early += first_stats.n_partials_folded_early;
                layer_stats_total.first_await_in_flight =
                    std::max(layer_stats_total.first_await_in_flight, first_stats.first_await_in_flight);
                layer_stats_total.first_await_recorded =
                    layer_stats_total.first_await_recorded || first_stats.first_await_recorded;
                layer_stats_total.max_in_flight = std::max(layer_stats_total.max_in_flight,
                                                           first_stats.max_in_flight);
                for (const worker_dispatch_stats & first_worker : first_stats.workers) {
                    for (worker_dispatch_stats & total_worker : layer_stats_total.workers) {
                        if (total_worker.endpoint != first_worker.endpoint) {
                            continue;
                        }
                        total_worker.ns_wait += first_worker.ns_wait;
                        total_worker.n_requests += first_worker.n_requests;
                        total_worker.n_experts_total += first_worker.n_experts_total;
                        break;
                    }
                }
            }
        }
        const dispatch_clock::time_point unpack_start =
            collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};
        if (ggml_is_contiguous(dst) && dst->type == GGML_TYPE_F32) {
            std::memcpy(dst->data, result.data(), result.size() * sizeof(float));
        } else {
            for (size_t i = 0; i < result.size(); ++i) {
                ggml_set_f32_1d(dst, (int) i, result[i]);
            }
        }
        if (trace_profile) {
            owner->spine_profile_wait_end(dispatch_clock::now());
        }
        if (context->chunk_count == 1 || context->chunk_index == context->chunk_count - 1) {
            owner->write_layer_trace(context->layer);
        }
        if (collect_stats) {
            const dispatch_clock::time_point total_end = dispatch_clock::now();
            owner->decode_ns_wait_ += layer_stats.ns_wait;
            owner->decode_ns_unpack_ += elapsed_ns(unpack_start, total_end);
            owner->decode_ns_total_ += layer_stats.ns_issue + layer_stats.ns_wait +
                                       elapsed_ns(unpack_start, total_end);
            if (context->chunk_count == 1 || context->chunk_index == context->chunk_count - 1) {
                ++owner->decode_layers_;
                owner->decode_ns_fold_overlapped_ += layer_stats_total.ns_fold_overlapped;
                owner->decode_n_partials_folded_early_ += layer_stats_total.n_partials_folded_early;
                owner->decode_first_await_in_flight_ += layer_stats_total.first_await_in_flight;
                owner->decode_max_in_flight_ = std::max(owner->decode_max_in_flight_,
                                                        layer_stats_total.max_in_flight);
            }
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
