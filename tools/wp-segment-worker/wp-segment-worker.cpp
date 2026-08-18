#include "wp-segment-worker.h"

#include "common.h"
#include "ggml-backend.h"
#include "llama-pipeline.h"
#include "llama-ext.h"
#include "log.h"
#include "pipeline-stage.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace wp_segment_worker {
namespace {

void send_error(pipe_channel::channel & channel, uint64_t seq_id, pipe_error_code code,
                const std::string & message) {
    channel.send_frame(PIPE_ERROR, seq_id, pipe_encode_error({ (uint32_t) code, message }));
    channel.flush();
}

bool matches_hello(const pipe_segment_hello & hello, const service_config & config) {
    return hello.segment_id == config.segment_id &&
        hello.layer_first == config.layer_first &&
        hello.layer_last == config.layer_last &&
        hello.model_identity_sha256 == config.model_identity_sha256 &&
        hello.n_embd == config.n_embd &&
        hello.n_vocab == config.n_vocab &&
        hello.wire_precision == PIPE_SEGMENT_WIRE_F32 &&
        (hello.capabilities & config.capabilities) == hello.capabilities;
}

// WP_SEGMENT_TAIL_LOGITS=1 restores the legacy logits-on-tail path (the A/B
// control arm). Must match the head's setting; a mismatch is caught at HELLO.
bool tail_logits_forced() {
    const char * value = std::getenv("WP_SEGMENT_TAIL_LOGITS");
    return value != nullptr && value[0] == '1';
}

class llama_runtime final : public runtime {
public:
    // send_logits is only ever true on the tail, and only under the legacy A/B arm.
    llama_runtime(llama_context * ctx, int32_t n_embd, int32_t output_width,
                  int32_t nextn_width, bool is_tail, bool send_logits,
                  std::vector<uint32_t> tap_layers) :
            ctx(ctx), n_embd(n_embd), output_width(output_width), nextn_width(nextn_width),
            is_tail(is_tail), send_logits(send_logits), tap_layers(std::move(tap_layers)) {}

    bool forward(const pipe_segment_fwd_req & request,
                 pipe_segment_fwd_resp & response,
                 uint32_t want_nextn_width,
                 std::string & error) override {
        if (request.n_seqs != 1 || request.n_pos_per_token == 0) {
            error = "segment worker only supports one sequence";
            return false;
        }
        if (request.positions.empty() || request.positions.front() != (int32_t) n_past_value) {
            error = "segment forward position does not continue the cache";
            return false;
        }
        for (uint32_t i = 0; i < request.n_tokens; ++i) {
            if (request.positions[(size_t) i * request.n_pos_per_token] != (int32_t) n_past_value + (int32_t) i) {
                error = "segment forward positions are not consecutive";
                return false;
            }
        }
        if (!llama_pipeline_stage_decode_hidden(
                ctx, n_embd, request.activations.data(), (int32_t) request.n_tokens,
                request.positions.data(), request.n_pos_per_token)) {
            error = "llama_decode failed";
            return false;
        }

        response.session_id = request.session_id;
        response.seq_id = request.seq_id;
        response.n_tokens = request.n_tokens;
        response.output_width = output_width;
        // The head's negotiated need, clamped by what this segment can actually produce.
        // A non-tail has nextn_width == 0 and can never be talked into sending one.
        const uint32_t capability = nextn_width > 0 ? (uint32_t) nextn_width : 0u;
        const uint32_t send_nextn = want_nextn_width < capability ? want_nextn_width : capability;
        response.nextn_width = send_nextn;
        response.nextn_aliased = 0;
        response.nextn.clear();
        // The `else` branch is taken by logits-on-head (the default) AND by every
        // non-terminal segment. On the TAIL, llama_get_embeddings() is result_norm:
        // the post-output_norm hidden state, i.e. exactly the tensor the LM head
        // would have consumed (qwen35.cpp sets res->t_embd from the output_norm
        // output, before the projection). Shipping that lets the head finish the
        // matmul from a bit-identical input at n_embd instead of n_vocab width.
        if (is_tail && send_logits) {
            // Legacy A/B arm: the tail owns the projection and ships n_vocab f32
            // per token (993 KB/token at n_vocab=248320).
            float * logits = llama_get_logits(ctx);
            if (logits == nullptr) {
                error = "llama_get_logits returned null";
                return false;
            }
            response.activations.assign(logits, logits + (size_t) request.n_tokens * output_width);
        } else if (!llama_pipeline_stage_read_hidden(ctx, (int32_t) request.n_tokens, n_embd,
                                                     response.activations)) {
            error = "llama_get_embeddings returned null";
            return false;
        }

        if (is_tail && send_nextn > 0) {
            const float * nextn = llama_get_embeddings_nextn(ctx);
            if (nextn == nullptr) {
                error = "llama_get_embeddings_nextn returned null";
                return false;
            }
            const size_t n = (size_t) request.n_tokens * send_nextn;

            // NEXTN WIRE DEDUP. Under the HIDDEN terminal payload this segment has just
            // written the post-output_norm hidden state into response.activations, and
            // for the production architecture t_h_nextn is THAT SAME TENSOR
            // (qwen35.cpp:350-361 sets t_h_nextn from the output_norm output and then
            // takes t_embd from it), so the two runs are bit-identical and the frame was
            // carrying 2 x n_embd f32 per token where one would do.
            //
            // This is decided by COMPARING, not by assuming. It is not universally true:
            // under the LOGITS arm `activations` is n_vocab logits and the sideband is
            // genuinely different data, and other architectures point t_h_nextn at a
            // confidence vector instead (deepseek4.cpp:373, dflash.cpp:418). The memcmp
            // covers every one of those without this worker having to know which model
            // it loaded, and it is ~80 KB per 4-token block against ~40 KB/token of wire
            // at the measured ~888 Mbps -- three orders of magnitude cheaper than the
            // bytes it removes. Reconstruction on the head is a plain copy, so the
            // sideband the drafter sees is bit-for-bit what it saw before.
            if (send_nextn == (uint32_t) output_width &&
                response.activations.size() == n &&
                std::memcmp(response.activations.data(), nextn, n * sizeof(float)) == 0) {
                response.nextn_aliased = 1;
            } else {
                response.nextn.assign(nextn, nextn + n);
            }
        }

        // INTERIOR TAPS: one [n_tokens, n_embd] block per requested layer, concatenated
        // in the ascending order negotiated at HELLO. The rows land in batch order,
        // which is the layout the head's embd_layer_inp buffer expects, so the head can
        // memcpy them straight in.
        if (!tap_layers.empty()) {
            response.tap_width = (uint32_t) n_embd;
            response.n_taps    = (uint32_t) tap_layers.size();
            response.taps.clear();
            response.taps.reserve((size_t) request.n_tokens * n_embd * tap_layers.size());

            for (const uint32_t lid : tap_layers) {
                const float * rows = llama_get_embeddings_layer_inp(ctx, lid);
                if (rows == nullptr) {
                    error = "llama_get_embeddings_layer_inp returned null for layer " + std::to_string(lid);
                    return false;
                }
                response.taps.insert(response.taps.end(), rows,
                                     rows + (size_t) request.n_tokens * n_embd);
            }
        }

        // WP_SEGMENT_TRACE=1: per-hop activation hashes for determinism bisection
        if (std::getenv("WP_SEGMENT_TRACE")) {
            auto fnv = [](const float * p, size_t n) {
                uint64_t h = 1469598103934665603ull;
                const uint8_t * b = (const uint8_t *) p;
                for (size_t i = 0; i < n * sizeof(float); ++i) { h = (h ^ b[i]) * 1099511628211ull; }
                return h;
            };
            fprintf(stderr, "WPTRACE seg fwd seq=%llu n=%u in=%016llx out=%016llx\n",
                    (unsigned long long) request.seq_id, request.n_tokens,
                    (unsigned long long) fnv(request.activations.data(), request.activations.size()),
                    (unsigned long long) fnv(response.activations.data(), response.activations.size()));
            fflush(stderr);
        }

        n_past_value += request.n_tokens;
        return true;
    }

    void reset() override {
        llama_memory_clear(llama_get_memory(ctx), true);
        n_past_value = 0;
    }

    bool trim(uint32_t value, std::string & error) override {
        if (value > n_past_value) {
            error = "trim point is beyond the current cache";
            return false;
        }
        if (value != n_past_value && !llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) value, -1)) {
            error = "cache cannot restore the requested recurrent snapshot";
            return false;
        }
        n_past_value = value;
        return true;
    }

    uint32_t n_past() const override {
        return n_past_value;
    }

private:
    llama_context * ctx;
    int32_t         n_embd;
    int32_t         output_width;
    int32_t         nextn_width;
    bool            is_tail;
    bool            send_logits;
    std::vector<uint32_t> tap_layers;
    uint32_t        n_past_value = 0;
};

void apply_manifest_devices(common_params & params, const pipe_dense_segment::segment & segment) {
    if (!params.devices.empty()) {
        return;
    }
    if (segment.devices.size() == 1 && segment.devices.front() == "none") {
        params.devices.push_back(nullptr);
        return;
    }

    ggml_backend_load_all();
    for (const std::string & device_name : segment.devices) {
        ggml_backend_dev_t device = ggml_backend_dev_by_name(device_name.c_str());
        if (device == nullptr || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            throw std::runtime_error("dense segment manifest names unavailable device '" + device_name + "'");
        }
        params.devices.push_back(device);
    }
    params.devices.push_back(nullptr);

    if (segment.split_mode == "tensor") {
        params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    }
    if (!segment.tensor_split.empty()) {
        if (segment.tensor_split.size() > std::size(params.tensor_split)) {
            throw std::runtime_error("dense segment manifest tensor_split has too many entries");
        }
        std::fill(std::begin(params.tensor_split), std::end(params.tensor_split), 0.0f);
        for (size_t i = 0; i < segment.tensor_split.size(); ++i) {
            params.tensor_split[i] = segment.tensor_split[i];
        }
    }
}

} // namespace

resolved_segment resolve_segment(const options & options) {
    if (options.manifest_path.empty()) {
        throw std::invalid_argument("--manifest is required");
    }
    const pipe_dense_segment::manifest manifest =
        pipe_dense_segment::load_manifest(options.manifest_path.string());
    const auto it = std::find_if(manifest.segments.begin(), manifest.segments.end(),
        [&](const pipe_dense_segment::segment & segment) { return segment.id == options.segment_id; });
    if (it == manifest.segments.end()) {
        throw std::runtime_error("dense segment manifest has no segment " + std::to_string(options.segment_id));
    }

    std::filesystem::path stage_gguf = it->stage_gguf;
    if (stage_gguf.is_relative()) {
        stage_gguf = options.manifest_path.parent_path() / stage_gguf;
    }
    return { manifest, *it, stage_gguf.lexically_normal() };
}

int serve_connection(pipe_channel::channel & channel, const service_config & config, runtime & worker_runtime) {
    bool hello_done = false;
    bool session_active = false;
    uint64_t session_id = 0;
    uint64_t cache_epoch = 0;
    // NEXTN SIDEBAND, negotiated at HELLO and fixed for the connection. 0 until then, so
    // a forward that somehow preceded the handshake could not ship one.
    uint32_t negotiated_nextn_width = 0;

    try {
        while (true) {
            pipe_channel::received_frame frame;
            if (!pipe_channel::channel::harvest({ &channel }, frame, -1)) {
                continue;
            }

            if (!hello_done) {
                if (frame.type != PIPE_SEGMENT_HELLO || frame.seq_id != 0) {
                    send_error(channel, frame.seq_id, PIPE_ERR_HELLO, "expected segment HELLO");
                    return 1;
                }
                pipe_segment_hello hello;
                try {
                    hello = pipe_decode_segment_hello(frame.payload.data(), frame.payload.size());
                } catch (const pipe_protocol_error & error) {
                    send_error(channel, frame.seq_id, error.code, error.what());
                    return 1;
                }
                // Only the TERMINAL segment is constrained: it decided at load time
                // whether the LM head is in its graph at all, so it cannot serve the
                // other kind. A middle segment returns hidden state either way and
                // simply echoes whatever the head negotiated.
                if (config.is_terminal && hello.terminal_kind != config.terminal_kind) {
                    // Name both sides: this is the one mismatch that would otherwise
                    // be a silent mis-slice of the terminal payload -- the head would
                    // read n_embd floats as if they were n_vocab logits.
                    send_error(channel, frame.seq_id, PIPE_ERR_HELLO,
                        "segment terminal kind mismatch: head requested " +
                        std::to_string(hello.terminal_kind) + ", this tail serves " +
                        std::to_string(config.terminal_kind) +
                        " (WP_SEGMENT_TAIL_LOGITS must match on head and tail)");
                    return 1;
                }
                // Interior taps are armed at LOAD time (llama_set_embeddings_layer_inp
                // changes the graph), so this segment can only serve the exact set it
                // was configured with. Unlike terminal_kind this binds EVERY segment,
                // terminal or not. Serving nothing when taps were asked for is the worst
                // available outcome: the head reads a stale buffer, conditions its draft
                // on it, and still emits bit-identical verified tokens, so no downstream
                // check -- including temp-0 parity -- could ever notice.
                if (hello.tap_layers != config.tap_layers) {
                    send_error(channel, frame.seq_id, PIPE_ERR_HELLO,
                        "segment interior tap mismatch: head requested " +
                        std::to_string(hello.tap_layers.size()) + " tap(s), this segment extracts " +
                        std::to_string(config.tap_layers.size()) +
                        " (head and worker must be given the same manifest)");
                    return 1;
                }
                if (!matches_hello(hello, config)) {
                    send_error(channel, frame.seq_id, PIPE_ERR_HELLO, "segment HELLO does not match the manifest");
                    return 1;
                }
                // NEXTN SIDEBAND. The head declares whether it will READ it; this
                // segment declares whether it can produce one. Only the intersection
                // goes on the wire, so a head that needs nothing pays nothing.
                //
                // Deliberately NOT a rejection when the head asks for a sideband this
                // segment has none of: only a TAIL produces nextn, and a head that
                // declares need to a middle hop is asking a question the hop is entitled
                // to answer "zero" to. The head cross-checks the ACK against its own
                // expectation, so a genuine disagreement -- a TAIL answering 0 to a head
                // that needs it -- still fails at the handshake, on the side that knows
                // what it asked for.
                negotiated_nextn_width = hello.nextn_need != 0 ? config.nextn_width : 0u;
                {
                    pipe_segment_hello_ack ack;
                    ack.version        = PIPE_SEGMENT_HELLO_VERSION;
                    ack.accepted       = true;
                    ack.n_vocab        = config.n_vocab;
                    ack.rs_snapshots   = config.recurrent_snapshots;
                    ack.terminal_kind  = hello.terminal_kind;
                    ack.terminal_width = config.output_width;
                    ack.tap_layers     = config.tap_layers;
                    ack.tap_width      = config.tap_layers.empty() ? 0u : (uint32_t) config.n_embd;
                    ack.nextn_width    = negotiated_nextn_width;
                    channel.send_frame(PIPE_SEGMENT_HELLO_ACK, 0, pipe_encode_segment_hello_ack(ack));
                }
                channel.flush();
                hello_done = true;
                continue;
            }

            if (frame.type == PIPE_PING) {
                channel.send_frame(PIPE_PONG, frame.seq_id, {});
                channel.flush();
                continue;
            }
            if (frame.type == PIPE_SEGMENT_CTRL) {
                pipe_segment_ctrl control;
                try {
                    control = pipe_decode_segment_ctrl(frame.payload.data(), frame.payload.size());
                } catch (const pipe_protocol_error & error) {
                    send_error(channel, frame.seq_id, error.code, error.what());
                    return 1;
                }

                pipe_segment_ctrl_ack ack;
                ack.control = control.control;
                ack.session_id = control.session_id;
                ack.cache_epoch = control.cache_epoch;
                ack.status = PIPE_SEGMENT_CTRL_APPLIED;
                if (control.control == PIPE_SEGMENT_CTRL_RESET) {
                    worker_runtime.reset();
                    session_active = true;
                    session_id = control.session_id;
                    cache_epoch = control.cache_epoch;
                    ack.n_past = 0;
                } else {
                    if (!session_active || control.session_id != session_id || control.cache_epoch != cache_epoch) {
                        send_error(channel, frame.seq_id, PIPE_ERR_STALE_SEQ, "stale segment control session or cache epoch");
                        return 1;
                    }
                    if (control.control == PIPE_SEGMENT_CTRL_KV_TRIM) {
                        std::string error;
                        if (!worker_runtime.trim(control.n_past, error)) {
                            ack.status = PIPE_SEGMENT_CTRL_MISS;
                        }
                        ack.n_past = worker_runtime.n_past();
                    } else {
                        ack.n_past = worker_runtime.n_past();
                    }
                }
                channel.send_frame(PIPE_SEGMENT_CTRL_ACK, frame.seq_id, pipe_encode_segment_ctrl_ack(ack));
                channel.flush();
                continue;
            }
            if (frame.type == PIPE_SEGMENT_FWD_REQ) {
                pipe_segment_fwd_req request;
                try {
                    request = pipe_decode_segment_fwd_req(frame.payload.data(), frame.payload.size(), config.n_embd);
                } catch (const pipe_protocol_error & error) {
                    send_error(channel, frame.seq_id, error.code, error.what());
                    return 1;
                }
                if (!session_active || request.session_id != session_id || request.seq_id != frame.seq_id) {
                    send_error(channel, frame.seq_id, PIPE_ERR_STALE_SEQ, "stale segment forward session or sequence");
                    return 1;
                }
                pipe_segment_fwd_resp response;
                std::string error;
                if (!worker_runtime.forward(request, response, negotiated_nextn_width, error)) {
                    send_error(channel, frame.seq_id, PIPE_ERR_DECODE, error);
                    return 1;
                }
                response.output_width = config.output_width;
                // The NEGOTIATION is authoritative for the sideband, not the config: the
                // config says what this segment could produce, the HELLO says what the
                // head will read. Re-stamped here so a runtime that ignored the argument
                // trips the encoder's length check instead of desynchronising the peer.
                response.nextn_width = negotiated_nextn_width;
                if (negotiated_nextn_width == 0) {
                    response.nextn.clear();
                    response.nextn_aliased = 0;
                }
                // config is authoritative for every width on the wire, as above.
                response.tap_width = config.tap_layers.empty() ? 0u : (uint32_t) config.n_embd;
                response.n_taps    = (uint32_t) config.tap_layers.size();
                channel.send_frame(PIPE_SEGMENT_FWD_RESP, frame.seq_id,
                    pipe_encode_segment_fwd_resp(response));
                channel.flush();
                continue;
            }

            send_error(channel, frame.seq_id, PIPE_ERR_BAD_FRAME, "unexpected segment frame");
            return 1;
        }
    } catch (const std::runtime_error & error) {
        if (std::string(error.what()).find("peer closed") != std::string::npos) {
            return 0;
        }
        LOG_ERR("wp-segment-worker: %s\n", error.what());
        return 1;
    }
}

// WP_SEGMENT_SELFTEST_NODES=1: per-node output hashing for determinism bisection.
static std::vector<std::pair<std::string, uint64_t>> g_selftest_nodes;
static bool g_selftest_capture = false;

static uint64_t selftest_fnv_bytes(const uint8_t * b, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h = (h ^ b[i]) * 1099511628211ull; }
    return h;
}

static bool selftest_eval_cb(ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) {
        return g_selftest_capture;
    }
    if (!g_selftest_capture) {
        return true;
    }
    const size_t nb = ggml_nbytes(t);
    uint64_t h = 0;
    if (nb > 0 && nb <= (size_t) 64 * 1024 * 1024) {
        std::vector<uint8_t> buf(nb);
        ggml_backend_tensor_get(t, buf.data(), 0, nb);
        h = selftest_fnv_bytes(buf.data(), nb);
    }
    g_selftest_nodes.emplace_back(std::string(ggml_get_name(t)) + "/" + ggml_op_name(t->op), h);
    return true;
}

int run(const options & options, common_params & params) {
    if (std::getenv("WP_SEGMENT_SELFTEST_NODES")) {
        params.cb_eval           = selftest_eval_cb;
        params.cb_eval_user_data = nullptr;
    }
    const resolved_segment resolved = resolve_segment(options);
    int32_t file_first = -1;
    int32_t file_last = -1;
    if (!llama_pipeline_peek_band_from_file(resolved.stage_gguf.string().c_str(), &file_first, &file_last) ||
        file_first != resolved.segment.layer_first || file_last != resolved.segment.layer_last) {
        LOG_ERR("wp-segment-worker: stage GGUF band does not match segment %u\n", resolved.segment.id);
        return 1;
    }
    params.model.path = resolved.stage_gguf.string();
    params.pipeline_layer_first = resolved.segment.layer_first;
    params.pipeline_layer_last = resolved.segment.layer_last;
    params.embedding = true;
    params.warmup = false;
    params.n_parallel = 1;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = options.recurrent_snapshots;
    apply_manifest_devices(params, resolved.segment);

    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("wp-segment-worker: failed to load %s\n", params.model.path.c_str());
        return 1;
    }
    if (llama_model_n_layer(model) != resolved.manifest.n_layer ||
        llama_model_n_embd(model) != resolved.manifest.n_embd) {
        LOG_ERR("wp-segment-worker: stage model dimensions do not match the manifest\n");
        return 1;
    }

    const bool is_tail = resolved.segment.layer_last == resolved.manifest.n_layer - 1;
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // Logits-on-head: by default the TAIL stops after output_norm and ships the
    // normed hidden state; the head does the projection. WP_SEGMENT_TAIL_LOGITS=1
    // keeps the LM head here for A/B. This is a LOAD-TIME decision because
    // llama_set_no_output_head() changes the graph shape.
    const bool tail_sends_logits = is_tail && tail_logits_forced();
    const uint32_t terminal_kind = tail_sends_logits
        ? (uint32_t) PIPE_SEGMENT_TERMINAL_LOGITS
        : (uint32_t) PIPE_SEGMENT_TERMINAL_HIDDEN;

    const int32_t output_width = tail_sends_logits ? n_vocab : resolved.manifest.n_embd;
    const int32_t nextn_width = is_tail ? llama_model_n_embd_out(model) : 0;
    if (is_tail && nextn_width > 0) {
        llama_set_embeddings_nextn(ctx, true, false);
    }
    if (is_tail && !tail_sends_logits) {
        llama_set_no_output_head(ctx, true);
    }
    if (is_tail) {
        LOG_INF("wp-segment-worker: tail terminal payload = %s (%d f32/token)\n",
                tail_sends_logits ? "LOGITS" : "HIDDEN (logits-on-head)", output_width);
    }

    // INTERIOR TAPS. Armed here, at LOAD time, for the same reason as
    // llama_set_no_output_head above: llama_set_embeddings_layer_inp() marks
    // t_layer_inp[il] a graph output and cannot be renegotiated once the graph is
    // reserved. The manifest already told us which layers, and the band check in
    // parse_manifest() has already guaranteed each one is inside this segment.
    for (const uint32_t lid : resolved.segment.tap_layers) {
        llama_set_embeddings_layer_inp(ctx, lid, true);
    }
    if (!resolved.segment.tap_layers.empty()) {
        std::string tap_list;
        for (const uint32_t lid : resolved.segment.tap_layers) {
            tap_list += (tap_list.empty() ? "" : ",") + std::to_string(lid);
        }
        LOG_INF("wp-segment-worker: interior taps = [%s] (%d f32/token each)\n",
                tap_list.c_str(), resolved.manifest.n_embd);
    }
    const uint64_t capabilities = PIPE_SEGMENT_CAP_FWD | PIPE_SEGMENT_CAP_RESET |
        PIPE_SEGMENT_CAP_KV_TRIM | PIPE_SEGMENT_CAP_PROMPT_REUSE |
        (llama_n_rs_seq(ctx) > 0 ? PIPE_SEGMENT_CAP_RECURRENT : 0);
    service_config config = {
        resolved.segment.id,
        resolved.segment.layer_first,
        resolved.segment.layer_last,
        resolved.manifest.model_identity_sha256,
        resolved.manifest.n_embd,
        (uint32_t) n_vocab,
        (uint32_t) output_width,
        (uint32_t) nextn_width,
        llama_n_rs_seq(ctx),
        capabilities,
        terminal_kind,
        is_tail,
        resolved.segment.tap_layers,
    };

    // WP_SEGMENT_SELFTEST=<n_tokens>: in-process determinism probe. Runs the SAME
    // synthetic activation batch K times (full reset in between) and prints the
    // output hash of each pass. No network, no peer segments. WP_SEGMENT_SELFTEST_K
    // sets the repeat count (default 4), WP_SEGMENT_SELFTEST_STEPS sets how many
    // single-token decodes follow the prompt pass (default 0).
    if (const char * selftest = std::getenv("WP_SEGMENT_SELFTEST")) {
        const int32_t n_tokens = std::max(1, atoi(selftest));
        const int     n_rep    = std::getenv("WP_SEGMENT_SELFTEST_K")
                                    ? std::max(1, atoi(std::getenv("WP_SEGMENT_SELFTEST_K"))) : 4;
        const int     n_steps  = std::getenv("WP_SEGMENT_SELFTEST_STEPS")
                                    ? std::max(0, atoi(std::getenv("WP_SEGMENT_SELFTEST_STEPS"))) : 0;
        const int32_t n_embd   = resolved.manifest.n_embd;

        // deterministic synthetic activations, stable across processes
        auto make_act = [&](int32_t base_pos, int32_t count) {
            std::vector<float> a((size_t) count * n_embd);
            for (int32_t t = 0; t < count; ++t) {
                uint32_t s = 0x9E3779B9u ^ (uint32_t) (base_pos + t);
                for (int32_t j = 0; j < n_embd; ++j) {
                    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                    a[(size_t) t * n_embd + j] = ((float) (s & 0xFFFF) / 32768.0f - 1.0f) * 0.5f;
                }
            }
            return a;
        };
        auto fnv = [](const float * p, size_t n) {
            uint64_t h = 1469598103934665603ull;
            const uint8_t * b = (const uint8_t *) p;
            for (size_t i = 0; i < n * sizeof(float); ++i) { h = (h ^ b[i]) * 1099511628211ull; }
            return h;
        };

        // No tap list on purpose, and it does NOT weaken the probe: run() already called
        // llama_set_embeddings_layer_inp() for the configured taps above, so this ctx
        // carries the tap-armed GRAPH -- which is the only part of the feature that could
        // perturb determinism. The runtime's list controls just the copy of the tap rows
        // into the response, and this probe calls probe.forward() directly, hashing
        // activations only; it never reaches serve_connection or the response encoder.
        // Passing the real list would add a per-pass copy nothing here reads.
        llama_runtime probe(ctx, n_embd, output_width, nextn_width, is_tail, tail_sends_logits, {});
        std::vector<std::vector<std::pair<std::string, uint64_t>>> per_rep;
        const bool capture_nodes = std::getenv("WP_SEGMENT_SELFTEST_NODES") != nullptr;
        for (int r = 0; r < n_rep; ++r) {
            probe.reset();
            g_selftest_nodes.clear();
            g_selftest_capture = capture_nodes;
            std::string line = "WPSELFTEST rep=" + std::to_string(r);
            for (int step = 0; step <= n_steps; ++step) {
                const int32_t base  = step == 0 ? 0 : n_tokens + step - 1;
                const int32_t count = step == 0 ? n_tokens : 1;
                const std::vector<float> act = make_act(base, count);
                std::vector<int32_t> pos(count);
                for (int32_t i = 0; i < count; ++i) { pos[i] = base + i; }

                pipe_segment_fwd_req  req;
                pipe_segment_fwd_resp resp;
                req.session_id = 1;
                req.seq_id = (uint64_t) step + 1;
                req.n_tokens = (uint32_t) count;
                req.n_seqs = 1;
                req.n_pos_per_token = 1;
                req.positions = pos;
                req.activations = act;

                std::string err;
                // Pass the segment's full capability width so the probe does exactly the
                // work the pre-negotiation build did -- the selftest is a determinism
                // bisection tool and its cost profile should not move with a wire change.
                if (!probe.forward(req, resp, nextn_width > 0 ? (uint32_t) nextn_width : 0u, err)) {
                    LOG_ERR("wp-segment-worker: selftest forward failed: %s\n", err.c_str());
                    return 1;
                }
                char buf[64];
                snprintf(buf, sizeof(buf), " s%d=%016llx", step,
                         (unsigned long long) fnv(resp.activations.data(), resp.activations.size()));
                line += buf;
            }
            g_selftest_capture = false;
            per_rep.push_back(g_selftest_nodes);
            fprintf(stderr, "%s nodes=%zu\n", line.c_str(), g_selftest_nodes.size());
            fflush(stderr);
        }
        if (capture_nodes) {
            for (size_t r = 1; r < per_rep.size(); ++r) {
                const auto & a = per_rep[r - 1];
                const auto & b = per_rep[r];
                if (a.size() != b.size()) {
                    fprintf(stderr, "WPSELFTEST diff rep%zu/rep%zu: node count %zu vs %zu\n",
                            r - 1, r, a.size(), b.size());
                    continue;
                }
                size_t n_diff = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                    if (a[i].second != b[i].second) {
                        if (n_diff < 8) {
                            fprintf(stderr, "WPSELFTEST diff rep%zu/rep%zu idx=%zu %s %016llx vs %016llx\n",
                                    r - 1, r, i, a[i].first.c_str(),
                                    (unsigned long long) a[i].second, (unsigned long long) b[i].second);
                        }
                        ++n_diff;
                    }
                }
                fprintf(stderr, "WPSELFTEST diff rep%zu/rep%zu total_diff=%zu of %zu\n",
                        r - 1, r, n_diff, a.size());
            }
            fflush(stderr);
        }
        return 0;
    }

    if (!pipe_transport_init()) {
        LOG_ERR("wp-segment-worker: failed to initialize TCP transport\n");
        return 1;
    }
    pipe_socket_ptr server = pipe_socket_t::create_server(resolved.segment.target.host.c_str(),
                                                            resolved.segment.target.port);
    if (!server) {
        LOG_ERR("wp-segment-worker: cannot listen on %s:%u\n", resolved.segment.target.host.c_str(),
                resolved.segment.target.port);
        return 1;
    }

    int rc = 0;
    while (true) {
        pipe_socket_ptr socket = server->accept();
        if (!socket) {
            rc = 1;
            break;
        }
        pipe_channel::channel channel(std::move(socket), "segment client");
        llama_runtime worker_runtime(ctx, resolved.manifest.n_embd, output_width, nextn_width,
                                     is_tail, tail_sends_logits, resolved.segment.tap_layers);
        rc = serve_connection(channel, config, worker_runtime);
        if (rc != 0) {
            break;
        }
    }
    pipe_transport_shutdown();
    return rc;
}

} // namespace wp_segment_worker
