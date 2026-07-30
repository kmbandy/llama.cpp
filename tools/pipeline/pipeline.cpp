// llama-pipeline: Phase 2 cross-machine pipeline stage driver.
//
// One process = one stage = one full llama.cpp instance, joined at a layer
// boundary over TCP (docs/dev/2026-07-28-pipeline-protocol-design.md). This
// is the LOOPBACK driver: stages run on 127.0.0.1 on different ports. F32 on
// the wire; F16 is negotiated in HELLO but not enabled here.
//
// Roles (the tool derives the role from the resolved band, it is not a flag):
//   head  (first == 0):                the DRIVER. Embeds, runs its band, reads
//                                      t_embd, sends FWD_REQ to the next stage,
//                                      blocks on PIPE_TOKEN, appends, repeats.
//   middle:                            hidden in, hidden out. FWD_REQ -> decode
//                                      -> FWD_RESP.
//   tail  (last == n_layer-1):         hidden in, logits, samples, PIPE_TOKEN.
//
// Middle/tail MUST run with warmup disabled (--no-warmup): a token-based
// warmup decode on a stage without token_embd fails loudly in
// llm_graph_input_hidden::set_input by design. That guard is correct and is
// not weakened here.
//
// Usage:
//   # tail (listens)
//   llama-pipeline -m tail.gguf --pipeline-listen 127.0.0.1:9001 --no-warmup
//   # head (drives), connects to the tail
//   llama-pipeline -m head.gguf --pipeline-peer 127.0.0.1:9001 \
//       -p "prompt" -n 32 --seed 0 --temp 0
//
// The stage band comes from the stage GGUF's pipeline.layer_first/last
// metadata (wp-stage-split) or an explicit --pipeline-layers FIRST-LAST.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include "pipe-protocol.h"
#include "pipe-transport.h"
#include "llama-pipeline.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#  include <signal.h>
#endif

namespace {

// ---------------------------------------------------------------------------
// endpoint parsing "host:port"

bool parse_endpoint(const std::string & s, std::string & host, int & port) {
    const size_t colon = s.rfind(':');
    if (colon == std::string::npos) {
        return false;
    }
    host = s.substr(0, colon);
    port = std::atoi(s.substr(colon + 1).c_str());
    return !host.empty() && port > 0 && port <= 65535;
}

// ---------------------------------------------------------------------------
// pipeline-specific args, parsed manually before common_params_parse so that
// common never sees an unknown flag. Everything else defers to common.

struct pipe_cli {
    std::string listen;   // --pipeline-listen host:port  (middle/tail)
    std::string peer;     // --pipeline-peer   host:port  (head/middle -> next)
};

// remove our flags from argv in place; returns false + prints on parse error
bool parse_pipe_args(int & argc, char ** argv, pipe_cli & pc) {
    int w = 1;
    for (int r = 1; r < argc; ++r) {
        const std::string a = argv[r];
        auto take = [&](std::string & dst, const char * name) {
            if (r + 1 >= argc) {
                LOG_ERR("%s needs a value\n", name);
                return false;
            }
            dst = argv[++r];
            return true;
        };
        if (a == "--pipeline-listen") {
            if (!take(pc.listen, "--pipeline-listen")) return false;
        } else if (a == "--pipeline-peer") {
            if (!take(pc.peer, "--pipeline-peer")) return false;
        } else {
            argv[w++] = argv[r];
        }
    }
    argc = w;
    return true;
}

// ---------------------------------------------------------------------------
// resolved stage context

struct stage_ctx {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;

    int32_t first   = -1;   // resolved band
    int32_t last    = -1;
    int32_t n_layer = 0;    // global
    int32_t n_embd  = 0;

    pipe_role role = PIPE_ROLE_HEAD;

    bool is_head()   const { return first == 0; }
    bool is_tail()   const { return last == n_layer - 1; }
};

void resolve_role(stage_ctx & s) {
    if (s.is_head()) {
        s.role = PIPE_ROLE_HEAD;
    } else if (s.is_tail()) {
        s.role = PIPE_ROLE_TAIL;
    } else {
        s.role = PIPE_ROLE_MIDDLE;
    }
}

// ---------------------------------------------------------------------------
// HELLO handshake (symmetric): send ours, recv theirs, validate the combined
// stage set. Throws pipe_protocol_error / returns false on failure. `stages`
// is the full ordered stage set this process believes the pipeline has.
//
// Phase 2 loopback is 2-stage (head+tail) or 3-stage; the caller assembles
// `stages` from its own band plus the peers it is connected to BEFORE calling
// handshake on each link.

bool do_handshake(pipe_socket_t & sock, const stage_ctx & s,
                  const std::vector<llama_pipeline_stage> & stages) {
    pipe_hello mine;
    mine.role        = (uint32_t) s.role;
    mine.layer_first = s.first;
    mine.layer_last  = s.last;
    mine.n_layer     = s.n_layer;
    mine.n_embd      = s.n_embd;
    mine.hidden_type = PIPE_HIDDEN_F32;
    mine.model_hash  = 0; // structural hash not yet computed for stage files

    std::vector<uint8_t> enc = pipe_encode_hello(mine);
    if (!pipe_send_frame(sock, PIPE_HELLO, 0, enc.data(), enc.size())) {
        LOG_ERR("pipe: failed to send HELLO\n");
        return false;
    }

    pipe_frame_type type;
    uint64_t seq_id;
    std::vector<uint8_t> payload;
    if (!pipe_recv_frame(sock, type, seq_id, payload)) {
        LOG_ERR("pipe: failed to receive HELLO\n");
        return false;
    }
    if (type != PIPE_HELLO) {
        LOG_ERR("pipe: expected HELLO, got frame type %u\n", (unsigned) type);
        return false;
    }

    pipe_hello peer;
    try {
        peer = pipe_decode_hello(payload.data(), payload.size());
        pipe_validate_hello(peer, s.n_layer, s.n_embd, PIPE_HIDDEN_F32, stages);
    } catch (const pipe_protocol_error & e) {
        LOG_ERR("pipe: HELLO rejected: %s\n", e.what());
        pipe_send_error(sock, 0, e.code, e.what());
        return false;
    }

    LOG_INF("pipe: HELLO ok: peer role=%u band [%d, %d] n_layer=%d n_embd=%d hidden=F32\n",
            peer.role, peer.layer_first, peer.layer_last, peer.n_layer, peer.n_embd);
    return true;
}

// ---------------------------------------------------------------------------
// build an embd-driven llama_batch for a middle/tail stage
//
// hidden: n_embd * n_tokens floats (F32 on the wire in Phase 2).
// pos/seq: from the FWD_REQ frame. All tokens marked as outputs so the middle
// can read t_embd for every token and the tail can read logits for the last.

void fill_embd_batch(llama_batch & batch, const pipe_fwd_req & req, int32_t n_embd,
                     const float * hidden) {
    const int32_t n = (int32_t) req.n_tokens;
    batch.n_tokens = n;

    // llama_batch_init allocated batch.embd with n_tokens*n_embd floats; copy
    // the received hidden into it (never repoint batch.embd -- llama_batch_free
    // frees the buffer llama_batch_init allocated).
    std::memcpy(batch.embd, hidden, (size_t) n * n_embd * sizeof(float));

    for (int32_t i = 0; i < n; ++i) {
        batch.pos     [i]    = req.pos[(size_t) i * req.n_pos_per_embd];
        batch.n_seq_id[i]    = 1;
        batch.seq_id  [i][0] = 0; // Phase 2: single sequence
        batch.logits  [i]    = 1; // all tokens are outputs
    }
}

// read the per-token hidden states out of a middle/head stage after decode.
// Requires cparams.embeddings = true and all tokens marked as outputs, so
// llama_get_embeddings returns n_tokens * n_embd contiguous floats.
bool read_hidden_out(llama_context * ctx, int32_t n_tokens, int32_t n_embd,
                     std::vector<float> & out) {
    const float * embd = llama_get_embeddings(ctx);
    if (embd == nullptr) {
        LOG_ERR("pipe: llama_get_embeddings returned null (embeddings ctx off?)\n");
        return false;
    }
    out.assign(embd, embd + (size_t) n_tokens * n_embd);
    return true;
}

// ---------------------------------------------------------------------------
// middle / tail serving loop: one FWD_REQ in flight, blocking

int run_stage_server(stage_ctx & s, pipe_socket_ptr server,
                     const std::vector<llama_pipeline_stage> & stages,
                     common_params & params) {
    LOG_INF("pipe: %s stage listening, band [%d, %d] of %d layers\n",
            s.role == PIPE_ROLE_MIDDLE ? "middle" : "tail", s.first, s.last, s.n_layer);

    pipe_socket_ptr client = server->accept();
    if (!client) {
        LOG_ERR("pipe: accept failed\n");
        return 1;
    }
    LOG_INF("pipe: client connected\n");

    if (!do_handshake(*client, s, stages)) {
        return 1;
    }

    // the tail owns lm_head and sampling. Build its sampler chain from the
    // params the (head) driver resolved; the head applies no sampling.
    common_sampler * smpl = nullptr;
    if (s.is_tail()) {
        smpl = common_sampler_init(s.model, params.sampling);
        if (smpl == nullptr) {
            LOG_ERR("pipe: failed to init tail sampler\n");
            return 1;
        }
    }

    int rc = 0;
    bool running = true;
    while (running) {
        pipe_frame_type type;
        uint64_t seq_id;
        std::vector<uint8_t> payload;
        if (!pipe_recv_frame(*client, type, seq_id, payload)) {
            LOG_ERR("pipe: connection closed or broken; dropping seq state\n");
            break;
        }

        switch (type) {
            case PIPE_PING:
                if (!pipe_send_frame(*client, PIPE_PONG, seq_id, nullptr, 0)) {
                    running = false;
                }
                continue;

            case PIPE_ERROR: {
                pipe_error e;
                try { e = pipe_decode_error(payload.data(), payload.size()); }
                catch (const pipe_protocol_error &) { e.code = 0; e.msg = "<unparseable>"; }
                LOG_ERR("pipe: peer sent ERROR code=%u: %s\n", e.code, e.msg.c_str());
                running = false;
                rc = 1;
                continue;
            }

            case PIPE_FWD_REQ:
                break; // handled below

            default:
                LOG_ERR("pipe: unexpected frame type %u on a stage server\n", (unsigned) type);
                pipe_send_error(*client, seq_id, PIPE_ERR_BAD_FRAME, "unexpected frame type");
                running = false;
                rc = 1;
                continue;
        }

        // ---- FWD_REQ ------------------------------------------------------
        pipe_fwd_req req;
        try {
            req = pipe_decode_fwd_req(payload.data(), payload.size(), s.n_embd, PIPE_HIDDEN_F32);
        } catch (const pipe_protocol_error & e) {
            pipe_send_error(*client, seq_id, e.code, e.what());
            running = false;
            rc = 1;
            break;
        }

        // F32 on the wire: interpret the hidden bytes directly as floats.
        const float * hidden = reinterpret_cast<const float *>(req.hidden.data());

        llama_batch batch = llama_batch_init((int32_t) req.n_tokens, s.n_embd, 1);
        fill_embd_batch(batch, req, s.n_embd, hidden);

        if (llama_decode(s.ctx, batch) != 0) {
            LOG_ERR("pipe: llama_decode failed on stage\n");
            pipe_send_error(*client, seq_id, PIPE_ERR_DECODE, "llama_decode failed on stage");
            llama_batch_free(batch);
            running = false;
            rc = 1;
            break;
        }
        llama_batch_free(batch);

        if (s.is_tail()) {
            // Only the LAST output row yields the next token. Sampling and
            // accepting every prompt row would corrupt the sampler's penalty
            // and grammar history with tokens that were never generated, so we
            // sample+accept exactly once (the last row). The wire contract
            // returns n_tokens ids (== FWD_REQ.n_tokens); the head consumes
            // only .back(), so the earlier slots carry the same id as padding.
            const int last_idx = (int) req.n_tokens - 1;
            const llama_token t = common_sampler_sample(smpl, s.ctx, last_idx);
            common_sampler_accept(smpl, t, true);

            pipe_token tok;
            tok.token_ids.assign(req.n_tokens, t);
            std::vector<uint8_t> out = pipe_encode_token(tok);
            if (!pipe_send_frame(*client, PIPE_TOKEN, seq_id, out.data(), out.size())) {
                running = false;
                rc = 1;
            }
        } else {
            // middle: return the band's hidden output for every token
            std::vector<float> hidden_out;
            if (!read_hidden_out(s.ctx, (int32_t) req.n_tokens, s.n_embd, hidden_out)) {
                pipe_send_error(*client, seq_id, PIPE_ERR_DECODE, "no embedding output on middle");
                running = false;
                rc = 1;
                break;
            }
            pipe_fwd_resp resp;
            resp.n_tokens = req.n_tokens;
            resp.hidden.resize(hidden_out.size() * sizeof(float));
            std::memcpy(resp.hidden.data(), hidden_out.data(), resp.hidden.size());
            std::vector<uint8_t> out = pipe_encode_fwd_resp(resp);
            if (!pipe_send_frame(*client, PIPE_FWD_RESP, seq_id, out.data(), out.size())) {
                running = false;
                rc = 1;
            }
        }
    }

    if (smpl) {
        common_sampler_free(smpl);
    }
    return rc;
}

// ---------------------------------------------------------------------------
// head driver: embed + own band, then drive the next stage

int run_head_driver(stage_ctx & s, pipe_socket_ptr next,
                    const std::vector<llama_pipeline_stage> & stages,
                    common_params & params) {
    if (!do_handshake(*next, s, stages)) {
        return 1;
    }

    // tokenize the prompt
    const llama_vocab * vocab = llama_model_get_vocab(s.model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> prompt =
        common_tokenize(vocab, params.prompt, /*add_special=*/add_bos, /*parse_special=*/true);
    if (prompt.empty()) {
        LOG_ERR("pipe: empty prompt after tokenize\n");
        return 1;
    }
    LOG_INF("pipe: head driving: %zu prompt tokens, predict %d\n",
            prompt.size(), params.n_predict);

    const int32_t n_ubatch = (int32_t) llama_n_ubatch(s.ctx);
    uint64_t seq_id = 1;
    llama_pos pos = 0;

    std::vector<llama_token> generated;

    // helper: push one ubatch of TOKENS through the head band, read t_embd at
    // the boundary, forward to the next stage, and return the sampled id(s).
    auto forward_tokens = [&](const llama_token * toks, int32_t n, llama_pos base_pos,
                              llama_token & out_last) -> bool {
        // 1) run the head band on a token batch (embeddings ctx on, all output)
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int32_t i = 0; i < n; ++i) {
            batch.token   [i]    = toks[i];
            batch.pos     [i]    = base_pos + i;
            batch.n_seq_id[i]    = 1;
            batch.seq_id  [i][0] = 0;
            batch.logits  [i]    = 1;
        }
        if (llama_decode(s.ctx, batch) != 0) {
            LOG_ERR("pipe: head llama_decode failed\n");
            llama_batch_free(batch);
            return false;
        }
        llama_batch_free(batch);

        // 2) read t_embd at the band boundary (all tokens, F32)
        std::vector<float> hidden;
        if (!read_hidden_out(s.ctx, n, s.n_embd, hidden)) {
            return false;
        }

        // 3) FWD_REQ to the next stage
        pipe_fwd_req req;
        req.n_tokens       = (uint32_t) n;
        req.n_pos_per_embd = 1;
        req.pos.reserve(n);
        for (int32_t i = 0; i < n; ++i) req.pos.push_back(base_pos + i);
        req.seq_tokens     = { n };
        req.hidden.resize(hidden.size() * sizeof(float));
        std::memcpy(req.hidden.data(), hidden.data(), req.hidden.size());

        std::vector<uint8_t> enc = pipe_encode_fwd_req(req, PIPE_HIDDEN_F32);
        if (!pipe_send_frame(*next, PIPE_FWD_REQ, seq_id, enc.data(), enc.size())) {
            LOG_ERR("pipe: FWD_REQ send failed\n");
            return false;
        }

        // 4) block on the reply: PIPE_TOKEN (from a tail) or FWD_RESP (from a
        //    middle; loopback 2-stage always has a tail next)
        pipe_frame_type rtype;
        uint64_t rseq;
        std::vector<uint8_t> rpayload;
        if (!pipe_recv_frame(*next, rtype, rseq, rpayload)) {
            LOG_ERR("pipe: connection broken awaiting reply (seq %llu); failing request loudly\n",
                    (unsigned long long) seq_id);
            return false;
        }
        if (rtype == PIPE_ERROR) {
            pipe_error e;
            try { e = pipe_decode_error(rpayload.data(), rpayload.size()); }
            catch (const pipe_protocol_error &) { e.code = 0; e.msg = "<unparseable>"; }
            LOG_ERR("pipe: next stage ERROR code=%u: %s\n", e.code, e.msg.c_str());
            return false;
        }
        if (rtype != PIPE_TOKEN) {
            LOG_ERR("pipe: expected PIPE_TOKEN from tail, got frame type %u\n", (unsigned) rtype);
            return false;
        }
        pipe_token tok;
        try {
            tok = pipe_decode_token(rpayload.data(), rpayload.size());
        } catch (const pipe_protocol_error & e) {
            LOG_ERR("pipe: bad PIPE_TOKEN: %s\n", e.what());
            return false;
        }
        if (tok.token_ids.empty()) {
            LOG_ERR("pipe: PIPE_TOKEN carried no ids\n");
            return false;
        }
        out_last = tok.token_ids.back();
        return true;
    };

    // ---- prompt processing (ubatch-sized FWD_REQs) ----
    llama_token last = -1;
    size_t done = 0;
    while (done < prompt.size()) {
        const int32_t n = (int32_t) std::min((size_t) n_ubatch, prompt.size() - done);
        if (!forward_tokens(prompt.data() + done, n, pos, last)) {
            return 1;
        }
        done += n;
        pos  += n;
        ++seq_id;
    }

    // the last prompt token's sampled id is the first generated token
    if (last < 0) {
        LOG_ERR("pipe: no token produced from prompt\n");
        return 1;
    }
    generated.push_back(last);

    const llama_token eos = llama_vocab_eos(vocab);
    if (last == eos && !params.sampling.ignore_eos) {
        // done immediately
    } else {
        // ---- autoregressive decode: 1-token FWD_REQs ----
        for (int32_t i = 1; i < params.n_predict; ++i) {
            llama_token next_tok;
            if (!forward_tokens(&last, 1, pos, next_tok)) {
                return 1;
            }
            ++pos;
            ++seq_id;
            generated.push_back(next_tok);
            last = next_tok;
            if (next_tok == eos && !params.sampling.ignore_eos) {
                break;
            }
        }
    }

    // Print the generated text directly to stdout with plain printf (NOT via
    // the log system) so that, combined with --log-disable --no-display-prompt,
    // stdout carries exactly the generated continuation and nothing else. This
    // is what the loopback harness diffs against the single-process reference.
    // A canonical token-id line goes to stderr as ground truth.
    std::string out_text;
    for (llama_token t : generated) {
        out_text += common_token_to_piece(s.ctx, t, /*special=*/true);
    }
    std::fprintf(stderr, "PIPELINE-TOKENS:");
    for (llama_token t : generated) {
        std::fprintf(stderr, " %d", (int) t);
    }
    std::fprintf(stderr, "\n");
    std::printf("%s", out_text.c_str());
    std::fflush(stdout);

    LOG_INF("pipe: generated %zu tokens over %llu pipeline requests\n",
            generated.size(), (unsigned long long) (seq_id - 1));
    return 0;
}

} // namespace

// satisfies -Wmissing-declarations
int llama_pipeline(int argc, char ** argv);

int llama_pipeline(int argc, char ** argv) {
    pipe_transport_init();

    pipe_cli pc;
    if (!parse_pipe_args(argc, argv, pc)) {
        return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION, nullptr)) {
        return 1;
    }

    if (pc.listen.empty() && pc.peer.empty()) {
        LOG_ERR("llama-pipeline: set --pipeline-listen (stage server) or --pipeline-peer (head driver)\n");
        return 1;
    }

    // Resolve the band now so we can pick the role and validate against the
    // model after load. Band comes from --pipeline-layers or the stage GGUF.
    const bool band_from_cli = llama_pipeline_band_enabled(
        params.pipeline_layer_first, params.pipeline_layer_last);

    // Middle/tail stages must not run a token-based warmup decode: a stage
    // without token_embd fails that warmup by design. Force it off unless the
    // stage turns out to be the head.
    if (band_from_cli && params.pipeline_layer_first != 0) {
        if (params.warmup) {
            LOG_WRN("pipe: non-head stage: forcing --no-warmup "
                    "(a token warmup on a stage without token_embd fails by design)\n");
        }
        params.warmup = false;
    }

    // The head and middle stages need per-token hidden output; the tail needs
    // logits. Enabling embeddings gives both (t_embd + t_logits); the tail
    // reads logits, the head/middle read t_embd. This is a context-level flag
    // and does not change the legacy single-process path (no band set).
    params.embedding = true;

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("pipe: failed to load model/context\n");
        return 1;
    }

    stage_ctx s;
    s.model   = model;
    s.ctx     = ctx;
    s.n_layer = llama_model_n_layer(model);
    s.n_embd  = llama_model_n_embd(model);

    // resolve the band: explicit flag, else stage-GGUF metadata (already
    // adopted into mparams by the loader), else the full range.
    llama_pipeline_stage band;
    try {
        band = llama_pipeline_resolve_band(
            params.pipeline_layer_first, params.pipeline_layer_last, s.n_layer);
    } catch (const std::runtime_error & e) {
        LOG_ERR("pipe: %s\n", e.what());
        return 1;
    }
    s.first = band.first;
    s.last  = band.last;
    resolve_role(s);

    // If the band came only from GGUF metadata (non-head) we could not have
    // forced --no-warmup above; common_init_from_params already ran a warmup.
    // That path is the operator's responsibility (pass --no-warmup); we only
    // guarantee the head warms up normally and warn here.
    if (!band_from_cli && s.first != 0 && params.warmup) {
        LOG_WRN("pipe: non-head stage adopted from GGUF ran a token warmup; "
                "this should have failed by design -- pass --no-warmup\n");
    }

    // The full stage set. Loopback 2-stage: head + tail. We build it from our
    // own band plus the peer we connect to / accept from. For the handshake we
    // need the complete set BEFORE validating; the loopback harness wires a
    // head [0,K-1] and a tail [K,n_layer-1], so the set is our band plus the
    // complementary band. The peer's HELLO supplies its band; we validate the
    // union. Here we assemble our own band and let the handshake fill the rest
    // by validating against a set that includes the peer's declared band.
    //
    // Concretely: validate the pair {ours, peer} once the peer's HELLO is in.
    // do_handshake is given a stage set; build the candidate set as ours plus
    // every other band the peer could claim is handled inside do_handshake via
    // the `stages` we pass. For the 2-stage loopback the correct set is:
    //   head: { [0,K-1], [K,n-1] }  (peer is the tail)
    //   tail: { [0,K-1], [K,n-1] }  (peer is the head)
    // We construct it as {ours, complement} where complement covers the rest of
    // [0, n_layer-1]; the peer's HELLO must match that complement exactly.
    std::vector<llama_pipeline_stage> stages;
    stages.push_back({ s.first, s.last });
    {
        llama_pipeline_stage comp;
        if (s.first == 0) {
            comp = { s.last + 1, s.n_layer - 1 };       // we are head; peer is the rest
        } else {
            comp = { 0, s.first - 1 };                  // we are tail; peer is the head side
        }
        stages.push_back(comp);
        // order them ascending for validate_stages
        if (stages[0].first > stages[1].first) {
            std::swap(stages[0], stages[1]);
        }
    }

    int rc = 1;
    if (!pc.peer.empty()) {
        // head driver (also used by a middle driving its next hop in N>2)
        std::string host; int port;
        if (!parse_endpoint(pc.peer, host, port)) {
            LOG_ERR("pipe: bad --pipeline-peer '%s' (want host:port)\n", pc.peer.c_str());
            return 1;
        }
        pipe_socket_ptr next = pipe_socket_t::connect(host.c_str(), port);
        if (!next) {
            LOG_ERR("pipe: connect to %s:%d failed\n", host.c_str(), port);
            return 1;
        }
        rc = run_head_driver(s, next, stages, params);
    } else {
        std::string host; int port;
        if (!parse_endpoint(pc.listen, host, port)) {
            LOG_ERR("pipe: bad --pipeline-listen '%s' (want host:port)\n", pc.listen.c_str());
            return 1;
        }
        pipe_socket_ptr server = pipe_socket_t::create_server(host.c_str(), port);
        if (!server) {
            LOG_ERR("pipe: listen on %s:%d failed\n", host.c_str(), port);
            return 1;
        }
        rc = run_stage_server(s, server, stages, params);
    }

    pipe_transport_shutdown();
    return rc;
}
