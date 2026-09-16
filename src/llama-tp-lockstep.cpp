// Cross-host tensor parallelism, milestone M3: LOCKSTEP.
//
// M1 made two processes agree on which rows of every tensor each of them owns; M2 gave them a
// connection and made the per-layer partial sums add up across it. Neither made the two processes
// run the same graph over the same batch, and without that the reduce frames of M2 are two ranks
// shouting past each other.
//
// This file is the leader/follower contract that closes it:
//
//   LEADER (rank 0, the process that owns world device 0) is an ordinary llama-server or
//   llama-cli. Every llama_decode()/llama_encode() it performs, and every llama_memory_* mutation
//   it makes, is mirrored to the follower first.
//
//   FOLLOWER (any other rank) has no sampler, no HTTP and no LM head rows. It loops on
//   llama_tp_follower_step(): receive one message, apply it, repeat.
//
// WHY MIRRORING AT llama_context::decode() IS THE RIGHT SEAM. Three candidate seams exist: the
// server's decode call sites, common/, and llama_decode itself.
//   - The server's call sites miss llama-cli, miss common_context_can_seq_rm()'s two-token probe,
//     and would have to be re-audited every time a decode is added.
//   - common/ misses everything that calls llama_decode directly.
//   - llama_context::decode() is the ONE funnel every batch passes through - prefill chunks, the
//     single-token decode step, and (later) the MTP verify batch, which is an ordinary token
//     batch and so needs no new message type at all (spec D.2b).
// And because the batch is mirrored BEFORE llama_batch_allocr splits it, the ubatch split is not
// shipped: it is DERIVED on both ranks from inputs that are already equal, exactly as M1 derived
// the row split from world state rather than shipping it. A 512-token prefill costs one 3 KB
// descriptor, not one per ubatch.
//
// WHY MIRRORING AT THE llama_memory_* C ENTRY POINTS IS THE RIGHT SEAM. Auditing this fork's
// server for KV mutations (see the M3 handoff) finds them behind common_memory::seq_rm/seq_cp/
// seq_add, behind llama-cli's context shift, and behind common_context_can_seq_rm() - three
// different layers, all of which bottom out in the six functions in llama-context.cpp's "memory"
// section. Hooking there catches all of them and cannot be forgotten by a future call site. The
// cost when TP is off is one load of a null global per call.
//
// DIVERGENCE. The two ranks share one monotonic operation counter. The leader stamps it into
// every DECODE and CTRL frame; the follower refuses anything that is not exactly its own counter
// plus one, and every frame carries an FNV-1a of its own body. A follower that detects divergence
// logs what it saw and stops, which closes the socket; the leader's next reduce or mirror then
// fails, ggml_backend_meta_graph_compute returns GGML_STATUS_FAILED, and llama_decode returns an
// error - a failed request, not a hung one.

#include "llama-context.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-vocab.h"

#include "pipeline/pipe-tp-comm.h"
#include "pipeline/pipe-tp-msg.h"

#include <cstring>
#include <cstdlib>

llama_context * g_llama_tp_leader = nullptr;

// ---------------------------------------------------------------------------------------------
// WP_TP_TRACE=1 - the lockstep trace
//
// The whole design rests on "both ranks perform the same operations, with the same arguments, in
// the same order, against the same memory state". Everything above enforces the ORDER (the
// operation counter) and the WIRE (the frame checksums), but nothing prints what was actually
// applied, and nothing prints what the memory module RETURNED. That last one matters: the
// llama_memory_* entry points that return bool have callers whose next action depends on the
// answer (the server's "could not partially remove, clear the whole sequence" fallback, and
// common_context_seq_rm()'s abort). Only the leader's return value is ever observed, so a rank
// that answers differently silently applies a different sequence of operations from there on.
//
// This trace is off unless WP_TP_TRACE=1, is read once, and every call site is already inside a
// branch that has established a TP world exists - so a run without --tp-world does not reach it.
// ---------------------------------------------------------------------------------------------

bool llama_tp_trace_enabled() {
    static const bool enabled = [] {
        const char * e = std::getenv("WP_TP_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    return enabled;
}

static const char * llama_tp_ctrl_name(uint8_t op) {
    switch (op) {
        case PIPE_TP_CTRL_MEM_CLEAR: return "mem_clear";
        case PIPE_TP_CTRL_SEQ_RM:    return "seq_rm";
        case PIPE_TP_CTRL_SEQ_CP:    return "seq_cp";
        case PIPE_TP_CTRL_SEQ_KEEP:  return "seq_keep";
        case PIPE_TP_CTRL_SEQ_ADD:   return "seq_add";
        case PIPE_TP_CTRL_SEQ_DIV:   return "seq_div";
        case PIPE_TP_CTRL_SHUTDOWN:  return "shutdown";
        default:                     return "?";
    }
}

void llama_tp_trace_mem(const char * role, uint8_t op,
                        int32_t a, int32_t b, int32_t c, int32_t d, uint8_t b0, int res) {
    if (!llama_tp_trace_enabled()) {
        return;
    }
    LLAMA_LOG_INFO("WP_TP_TRACE %s mem %-9s a=%d b=%d c=%d d=%d b0=%u -> %s\n",
            role, llama_tp_ctrl_name(op), a, b, c, d, (unsigned) b0,
            res < 0 ? "void" : (res ? "true" : "false"));
}

void llama_context::tp_trace_decode(bool is_encode, uint32_t n_tokens_all, uint32_t n_outputs_all,
                                    uint32_t n_ubatches, llama_pos pos_first, llama_pos pos_last) const {
    if (!llama_tp_trace_enabled()) {
        return;
    }
    const uint64_t n_exchanges = tp_comm ? tp_comm->stats().n_exchanges : 0;
    LLAMA_LOG_INFO("WP_TP_TRACE %s %s op_seq=%u n_tokens=%u n_outputs=%u n_ubatch=%u "
                   "pos=[%d,%d] exchanges=%llu\n",
            tp_is_rank0 ? "leader  " : "follower", is_encode ? "encode" : "decode",
            tp_op_seq, n_tokens_all, n_outputs_all, n_ubatches, pos_first, pos_last,
            (unsigned long long) n_exchanges);
}

// ---------------------------------------------------------------------------------------------
// HELLO
// ---------------------------------------------------------------------------------------------

bool llama_context::tp_hello_exchange(uint32_t n_world, uint32_t rank_first, uint32_t n_local,
                                      uint32_t type_k, uint32_t type_v) {
    if (!tp_comm) {
        return false;
    }

    const auto & hparams = model.hparams;

    pipe_tp_hello mine = {};
    mine.version    = PIPE_TP_PROTO_VERSION;
    mine.n_world    = n_world;
    mine.rank_first = rank_first;
    mine.n_local    = n_local;
    mine.n_ctx      = cparams.n_ctx;
    mine.n_ctx_seq  = cparams.n_ctx_seq;
    mine.n_rs_seq   = cparams.n_rs_seq;
    mine.n_batch    = cparams.n_batch;
    mine.n_ubatch   = cparams.n_ubatch;
    mine.n_seq_max  = cparams.n_seq_max;
    mine.type_k     = type_k;
    mine.type_v     = type_v;
    mine.n_vocab    = (uint32_t) model.vocab.n_tokens();
    mine.n_embd     = (uint32_t) hparams.n_embd;
    mine.n_layer    = (uint32_t) hparams.n_layer();
    mine.flags =
        (cparams.kv_unified ? PIPE_TP_HELLO_F_KV_UNIFIED  : 0) |
        (cparams.flash_attn ? PIPE_TP_HELLO_F_FLASH_ATTN  : 0) |
        (cparams.embeddings ? PIPE_TP_HELLO_F_EMBEDDINGS  : 0) |
        (cparams.causal_attn? PIPE_TP_HELLO_F_CAUSAL_ATTN : 0);

    const std::string desc = model.desc();
    mine.model_hash = pipe_tp_model_hash(desc.c_str(), model.n_elements(), mine.n_vocab, mine.n_layer);

    // The tensor split is what turns -ts into the world row map. Two ranks given different -ts
    // compute different maps and every reduce afterwards adds unrelated rows together, which
    // produces a plausible-looking wrong answer rather than a crash - so it is checked here.
    const float * ts = model.tensor_split();
    mine.split_hash = ts ? pipe_tp_fnv1a(ts, sizeof(float) * n_world) : 0;

    if (!tp_comm->send_msg(PIPE_TP_MSG_HELLO, &mine, sizeof(mine))) {
        LLAMA_LOG_ERROR("%s: cross-host TP: failed to send HELLO to the peer rank\n", __func__);
        return false;
    }

    uint8_t type = 0;
    if (!tp_comm->recv_msg(&type, tp_msg_buf)) {
        LLAMA_LOG_ERROR("%s: cross-host TP: failed to receive the peer's HELLO\n", __func__);
        return false;
    }
    if (type != PIPE_TP_MSG_HELLO || tp_msg_buf.size() != sizeof(pipe_tp_hello)) {
        LLAMA_LOG_ERROR("%s: cross-host TP: expected a HELLO of %zu bytes, got type %u of %zu bytes\n",
                __func__, sizeof(pipe_tp_hello), (unsigned) type, tp_msg_buf.size());
        return false;
    }

    pipe_tp_hello theirs = {};
    memcpy(&theirs, tp_msg_buf.data(), sizeof(theirs));

    const std::string bad = pipe_tp_hello_mismatch(mine, theirs);
    if (!bad.empty()) {
        LLAMA_LOG_ERROR("%s: cross-host TP: the two ranks are not configured for the same run: %s\n"
                        "%s: both ranks must be given the same model, the same -ts, the same "
                        "--tp-world and the same context settings\n",
                __func__, bad.c_str(), __func__);
        return false;
    }

    LLAMA_LOG_INFO("%s: cross-host TP: HELLO accepted (world %u devices, this rank [%u,%u), "
                   "peer [%u,%u))\n", __func__, n_world, rank_first, rank_first + n_local,
                   theirs.rank_first, theirs.rank_first + theirs.n_local);
    return true;
}

// ---------------------------------------------------------------------------------------------
// leader
// ---------------------------------------------------------------------------------------------

bool llama_context::tp_mirror_batch(const llama_batch & batch, bool is_encode) {
    if (!tp_comm || !tp_is_rank0) {
        return true; // not a leader: nothing to mirror
    }
    if (tp_dead) {
        LLAMA_LOG_ERROR("%s: cross-host TP: the peer rank is gone; failing this batch\n", __func__);
        return false;
    }

    std::string err;
    if (!pipe_tp_encode_batch(batch, is_encode, tp_op_seq + 1, tp_msg_buf, &err)) {
        LLAMA_LOG_ERROR("%s: cross-host TP: cannot mirror this batch: %s\n", __func__, err.c_str());
        tp_dead = true;
        return false;
    }
    if (!tp_comm->send_msg(PIPE_TP_MSG_DECODE, tp_msg_buf.data(), tp_msg_buf.size())) {
        LLAMA_LOG_ERROR("%s: cross-host TP: lost the peer rank while sending batch %u\n",
                __func__, tp_op_seq + 1);
        tp_dead = true;
        return false;
    }
    tp_op_seq++;
    return true;
}

bool llama_context::tp_mirror_ctrl(uint8_t op, int32_t a, int32_t b, int32_t c, int32_t d, uint8_t b0) {
    if (!tp_comm || !tp_is_rank0 || tp_dead) {
        return !tp_dead;
    }
    pipe_tp_ctrl ctrl;
    ctrl.op_seq = tp_op_seq + 1;
    ctrl.op     = op;
    ctrl.b0     = b0;
    ctrl.a = a; ctrl.b = b; ctrl.c = c; ctrl.d = d;

    pipe_tp_encode_ctrl(ctrl, tp_msg_buf);
    if (!tp_comm->send_msg(PIPE_TP_MSG_CTRL, tp_msg_buf.data(), tp_msg_buf.size())) {
        LLAMA_LOG_ERROR("%s: cross-host TP: lost the peer rank while sending control op %u\n",
                __func__, (unsigned) op);
        tp_dead = true;
        return false;
    }
    tp_op_seq++;
    return true;
}

void llama_tp_mirror_memory(llama_memory_t mem, uint8_t op,
                            int32_t a, int32_t b, int32_t c, int32_t d, uint8_t b0) {
    llama_context * ctx = g_llama_tp_leader;
    if (ctx == nullptr || mem == nullptr || mem != ctx->tp_memory()) {
        // Not the leader's own memory. A draft context's memory lands here and is deliberately
        // NOT mirrored: under spec E.3 Option B the follower has no draft context and no idea
        // speculation is happening.
        return;
    }
    ctx->tp_mirror_ctrl(op, a, b, c, d, b0);
}

// ---------------------------------------------------------------------------------------------
// follower
// ---------------------------------------------------------------------------------------------

int llama_context::tp_follower_step() {
    if (!tp_comm || tp_is_rank0) {
        LLAMA_LOG_ERROR("%s: called on a context that is not a tensor-parallel follower\n", __func__);
        return LLAMA_TP_STEP_ERROR;
    }

    uint8_t type = 0;
    if (!tp_comm->recv_msg(&type, tp_msg_buf)) {
        LLAMA_LOG_ERROR("%s: cross-host TP: lost the leader rank after operation %u\n",
                __func__, tp_op_seq);
        return LLAMA_TP_STEP_ERROR;
    }

    std::string err;

    if (type == PIPE_TP_MSG_ERROR) {
        LLAMA_LOG_ERROR("%s: cross-host TP: the leader rank aborted: %.*s\n", __func__,
                (int) tp_msg_buf.size(), (const char *) tp_msg_buf.data());
        return LLAMA_TP_STEP_ERROR;
    }

    if (type == PIPE_TP_MSG_CTRL) {
        pipe_tp_ctrl ctrl;
        if (!pipe_tp_decode_ctrl(tp_msg_buf.data(), tp_msg_buf.size(), &ctrl, &err)) {
            LLAMA_LOG_ERROR("%s: cross-host TP: bad CTRL frame after operation %u: %s\n",
                    __func__, tp_op_seq, err.c_str());
            return LLAMA_TP_STEP_ERROR;
        }
        if (ctrl.op == PIPE_TP_CTRL_SHUTDOWN) {
            LLAMA_LOG_INFO("%s: cross-host TP: the leader rank closed the world after %u operations\n",
                    __func__, tp_op_seq);
            return LLAMA_TP_STEP_SHUTDOWN;
        }
        if (ctrl.op_seq != tp_op_seq + 1) {
            LLAMA_LOG_ERROR("%s: cross-host TP: LOCKSTEP DIVERGENCE - expected operation %u, the "
                            "leader sent %u. The ranks are no longer running the same sequence of "
                            "calls; stopping rather than computing a wrong answer.\n",
                    __func__, tp_op_seq + 1, ctrl.op_seq);
            return LLAMA_TP_STEP_ERROR;
        }
        tp_op_seq = ctrl.op_seq;

        llama_memory_i * mem = memory.get();
        if (mem == nullptr) {
            LLAMA_LOG_ERROR("%s: cross-host TP: received a memory control op but this context has "
                            "no memory\n", __func__);
            return LLAMA_TP_STEP_ERROR;
        }
        // Applied through llama_memory_i directly, not through the llama_memory_* C API: the
        // follower must never re-enter the mirror (it is not the leader, so it would be a no-op,
        // but the intent is worth being explicit about).
        int trace_res = -1; // -1 = the op returns void
        switch (ctrl.op) {
            case PIPE_TP_CTRL_MEM_CLEAR: mem->clear(ctrl.b0 != 0);                            break;
            case PIPE_TP_CTRL_SEQ_RM:    trace_res = mem->seq_rm(ctrl.a, ctrl.b, ctrl.c) ? 1 : 0; break;
            case PIPE_TP_CTRL_SEQ_CP:    mem->seq_cp  (ctrl.a, ctrl.b, ctrl.c, ctrl.d);       break;
            case PIPE_TP_CTRL_SEQ_KEEP:  mem->seq_keep(ctrl.a);                               break;
            case PIPE_TP_CTRL_SEQ_ADD:   mem->seq_add (ctrl.a, ctrl.b, ctrl.c, ctrl.d);       break;
            case PIPE_TP_CTRL_SEQ_DIV:   mem->seq_div (ctrl.a, ctrl.b, ctrl.c, ctrl.d);       break;
            default:
                LLAMA_LOG_ERROR("%s: cross-host TP: unhandled control op %u\n", __func__, (unsigned) ctrl.op);
                return LLAMA_TP_STEP_ERROR;
        }
        llama_tp_trace_mem("follower", ctrl.op, ctrl.a, ctrl.b, ctrl.c, ctrl.d, ctrl.b0, trace_res);
        return LLAMA_TP_STEP_OK;
    }

    if (type != PIPE_TP_MSG_DECODE) {
        LLAMA_LOG_ERROR("%s: cross-host TP: unexpected message type %u after operation %u\n",
                __func__, (unsigned) type, tp_op_seq);
        return LLAMA_TP_STEP_ERROR;
    }

    pipe_tp_batch_desc desc;
    if (!pipe_tp_decode_batch(tp_msg_buf.data(), tp_msg_buf.size(), &desc, &err)) {
        LLAMA_LOG_ERROR("%s: cross-host TP: bad DECODE frame after operation %u: %s\n",
                __func__, tp_op_seq, err.c_str());
        return LLAMA_TP_STEP_ERROR;
    }
    if (desc.op_seq != tp_op_seq + 1) {
        LLAMA_LOG_ERROR("%s: cross-host TP: LOCKSTEP DIVERGENCE - expected operation %u, the "
                        "leader sent batch %u. The ranks are no longer running the same sequence "
                        "of calls; stopping rather than computing a wrong answer.\n",
                __func__, tp_op_seq + 1, desc.op_seq);
        return LLAMA_TP_STEP_ERROR;
    }
    tp_op_seq = desc.op_seq;

    llama_batch batch = desc.view();
    const int ret = desc.is_encode ? encode(batch) : decode(batch);
    if (ret != 0) {
        // Both ranks run the same batch against the same memory state, so a failure here is
        // either a real out-of-memory (which the leader will hit too) or a divergence. Either
        // way the follower cannot continue: stop, which drops the socket and fails the leader's
        // request instead of leaving it blocked on a reduce that will never arrive.
        LLAMA_LOG_ERROR("%s: cross-host TP: %s of mirrored batch %u (%d tokens) returned %d; "
                        "stopping the follower\n", __func__,
                desc.is_encode ? "encode" : "decode", desc.op_seq, batch.n_tokens, ret);
        return LLAMA_TP_STEP_ERROR;
    }

    // llama_context::decode() deliberately does NOT wait for the graph it just submitted: the
    // comment at its tail says so ("wait for the computation to finish (automatically done when
    // obtaining the model output)"). On the leader that wait is paid for by the very next thing
    // the server does - llama_get_logits_ith() calls llama_context::synchronize(). The follower
    // used to get it the same way, through the logits/embeddings readback. f80933d61 removed that
    // readback (the follower owns no LM head rows and cannot gather them), and with it the only
    // synchronize() on this rank: from that commit on the follower returned to the message loop
    // with the ubatch's graph still in flight and then applied the NEXT frame - a memory mutation
    // or a fresh graph build - on top of it. Restore the wait here, where the readback used to be.
    synchronize();

    return LLAMA_TP_STEP_OK;
}

// ---------------------------------------------------------------------------------------------
// public C API
// ---------------------------------------------------------------------------------------------

bool llama_tp_should_listen(const char * peer, int32_t rank_first, int32_t tp_listen) {
    if (tp_listen >= 0) {
        return tp_listen != 0; // the operator said so
    }
    // AUTO. Rank 0 is the natural listener - that is what every launch line before the firewall
    // problem used - but only if tp_peer actually names an address this machine can bind. An
    // address it cannot bind is unambiguously a "dial there", and treating it as one is what lets
    // rank 0 connect out to a listening rank 1 without anyone having to pass --tp-listen 0.
    if (peer == nullptr || peer[0] == '\0') {
        return false;
    }
    std::string host;
    int port = 0;
    if (!pipe_tp_comm::parse_peer(peer, &host, &port)) {
        return false;
    }
    return rank_first == 0 && pipe_tp_comm::host_is_local(host);
}

bool llama_tp_prebind_peer(const char * peer) {
    if (peer == nullptr || peer[0] == '\0') {
        return false;
    }
    std::string host;
    int port = 0;
    if (!pipe_tp_comm::parse_peer(peer, &host, &port)) {
        LLAMA_LOG_ERROR("%s: invalid --tp-peer '%s', expected host:port\n", __func__, peer);
        return false;
    }
    if (!pipe_tp_comm::prebind(host, port)) {
        LLAMA_LOG_ERROR("%s: cross-host TP: could not bind %s:%d\n", __func__, host.c_str(), port);
        return false;
    }
    LLAMA_LOG_INFO("%s: cross-host TP: listening on %s:%d BEFORE the model load, so the peer rank "
                   "can connect while this rank is still loading\n", __func__, host.c_str(), port);
    return true;
}

bool llama_tp_is_follower(const llama_context * ctx) {
    return ctx != nullptr && ctx->tp_is_follower();
}

int32_t llama_tp_follower_step(llama_context * ctx) {
    if (ctx == nullptr) {
        return LLAMA_TP_STEP_ERROR;
    }
    return (int32_t) ctx->tp_follower_step();
}
