// Cross-host tensor parallelism, M3: the lockstep loop, over a real loopback socket.
//
// test-tp-msg proves the codec round-trips. This proves the thing the codec exists for: that a
// LEADER emitting a realistic sequence of batches and memory mutations, interleaved with the M2
// reduce exchanges on the SAME connection, is applied by a FOLLOWER in exactly the same order,
// with exactly the same arguments.
//
// The follower here is a faithful stand-in for llama_context::tp_follower_step() - same message
// routing, same operation-counter check, same dispatch table - writing each applied operation into
// a call log instead of into a KV cache, because a real one needs a model and a GPU. The leader
// writes the operation it INTENDED into its own log. The two logs must be identical.
//
// The interleaving is the point. M2 gave the two ranks one connection carrying REDUCE frames; M3
// puts DECODE and CTRL frames on the same wire. Nothing separates them but ordering, so a test
// that sends control frames without any reduces between them would not exercise the property that
// actually has to hold.

#include "pipe-tp-comm.h"
#include "pipe-tp-msg.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static const char * TEST_HOST = "127.0.0.1";
static int          TEST_PORT = 47841;

// ---------------------------------------------------------------------------------------------
// the script both ranks must agree on
// ---------------------------------------------------------------------------------------------

struct op_batch {
    std::vector<llama_token>  token;
    std::vector<llama_pos>    pos;
    std::vector<int32_t>      n_seq_id;
    std::vector<llama_seq_id> seq_flat;
    std::vector<int8_t>       logits;
    bool with_pos = true, with_seq = true, with_logits = true;
    bool is_encode = false;
};

struct script_op {
    enum kind_t { BATCH, CTRL, REDUCE } kind;
    op_batch     batch;              // BATCH
    pipe_tp_ctrl ctrl;               // CTRL (op_seq filled in by the leader)
    size_t       n_values = 0;       // REDUCE
};

static op_batch make_batch(int n, int pos0, bool with_pos, bool with_seq, bool with_logits) {
    op_batch b;
    b.with_pos = with_pos; b.with_seq = with_seq; b.with_logits = with_logits;
    for (int i = 0; i < n; i++) {
        b.token.push_back(1000 + (pos0 + i) * 13);
        b.pos.push_back(pos0 + i);
        b.n_seq_id.push_back(1);
        b.seq_flat.push_back(0);
        b.logits.push_back(i == n - 1 ? 1 : 0);
    }
    return b;
}

// The sequence a real request produces: a warm-up decode and its memory clear, a chunked prefill
// with reduces between the chunks, a run of single-token decodes, a prompt-cache truncation, a
// cache-reuse shift (seq_rm + seq_add, the pair tools/server issues at server-context.cpp:5316),
// a slot copy, and a speculative rollback.
static std::vector<script_op> build_script() {
    std::vector<script_op> s;

    auto add_batch = [&](op_batch b) {
        script_op o; o.kind = script_op::BATCH; o.batch = std::move(b); s.push_back(std::move(o));
    };
    auto add_ctrl = [&](uint8_t op, int32_t a, int32_t b, int32_t c, int32_t d, uint8_t b0) {
        script_op o; o.kind = script_op::CTRL;
        o.ctrl.op = op; o.ctrl.a = a; o.ctrl.b = b; o.ctrl.c = c; o.ctrl.d = d; o.ctrl.b0 = b0;
        s.push_back(o);
    };
    auto add_reduce = [&](size_t n) {
        script_op o; o.kind = script_op::REDUCE; o.n_values = n; s.push_back(o);
    };

    // warm-up: llama_batch_get_one leaves every optional array null
    { op_batch w; w.token = {1, 2}; w.with_pos = w.with_seq = w.with_logits = false; add_batch(w); }
    add_reduce(8);
    add_ctrl(PIPE_TP_CTRL_MEM_CLEAR, 0, 0, 0, 0, 1);

    // prefill in two 512-token chunks, with a reduce inside each
    add_batch(make_batch(512, 0, true, true, true));
    add_reduce(4096);
    add_batch(make_batch(512, 512, true, true, true));
    add_reduce(4096);

    // decode steps
    for (int i = 0; i < 8; i++) {
        add_batch(make_batch(1, 1024 + i, true, true, true));
        add_reduce(64);
    }

    // prompt-cache truncation, then the cache-reuse shift pair
    add_ctrl(PIPE_TP_CTRL_SEQ_RM,  0, 900, -1, 0, 0);
    add_ctrl(PIPE_TP_CTRL_SEQ_RM,  0, 100, 200, 0, 0);
    add_ctrl(PIPE_TP_CTRL_SEQ_ADD, 0, 200, 900, -100, 0);

    // parallel-slot copy and a keep
    add_ctrl(PIPE_TP_CTRL_SEQ_CP,   0, 1, -1, -1, 0);
    add_ctrl(PIPE_TP_CTRL_SEQ_KEEP, 1, 0, 0, 0, 0);

    // a speculative verify batch (an ordinary multi-token batch) and the rollback after a partial
    // acceptance - spec D.2b: the follower has no idea speculation is happening
    add_batch(make_batch(3, 1032, true, true, true));
    add_reduce(192);
    add_ctrl(PIPE_TP_CTRL_SEQ_RM, 0, 1034, -1, 0, 0);

    // context shift
    add_ctrl(PIPE_TP_CTRL_SEQ_DIV, 0, 0, 512, 2, 0);

    // THE REQUEST BOUNDARY: decode, seq_rm, decode.
    //
    // This is the sequence tools/server/server-context.cpp performs between two completions on
    // the same slot, and the one the 2026-09-05 live run answered correctly the first time and
    // with an immediate EOS every time after. The server keeps_first(n_past), takes
    // p0 = prompt.tokens.pos_next() and calls seq_rm(slot, p0, -1); when the model can only roll
    // the recurrent state back by n_rs_seq tokens, n_past collapses to 0 and p0 with it, so the
    // op that actually crosses the wire is seq_rm(seq, 0, -1) - a FULL removal expressed as a
    // range, not as (-1, -1). Both spellings are scripted here because llama_memory_recurrent
    // treats them differently on the way in (only the p0 == 0 form takes the rm_all branch that
    // resets the rollback index) and a mirror that normalised one into the other would desync.
    add_ctrl(PIPE_TP_CTRL_SEQ_RM, 0, 0, -1, 0, 0);
    add_batch(make_batch(32, 0, true, true, true));   // request 2's prompt, reprocessed from pos 0
    add_reduce(2048);
    add_batch(make_batch(1, 32, true, true, true));   // and its first generated token
    add_reduce(64);
    add_ctrl(PIPE_TP_CTRL_SEQ_RM, 0, -1, -1, 0, 0);   // request 3 starts from a cleared sequence
    add_batch(make_batch(28, 0, true, true, true));
    add_reduce(1792);
    return s;
}

// ---------------------------------------------------------------------------------------------
// call logs
// ---------------------------------------------------------------------------------------------

// A one-line, fully-specified record of an applied operation. Deliberately includes the operation
// counter and a hash of the batch contents: two logs matching means the ranks applied the same
// operations, with the same arguments, in the same order, at the same counter.
static std::string log_batch(uint32_t op_seq, const llama_batch & b, bool is_encode) {
    char buf[256];
    uint64_t h = 0xcbf29ce484222325ull;
    h = pipe_tp_fnv1a(b.token, sizeof(llama_token) * (size_t) b.n_tokens, h);
    if (b.pos) {
        h = pipe_tp_fnv1a(b.pos, sizeof(llama_pos) * (size_t) b.n_tokens, h);
    }
    if (b.logits) {
        h = pipe_tp_fnv1a(b.logits, (size_t) b.n_tokens, h);
    }
    if (b.n_seq_id) {
        for (int32_t i = 0; i < b.n_tokens; i++) {
            h = pipe_tp_fnv1a(&b.n_seq_id[i], sizeof(int32_t), h);
            h = pipe_tp_fnv1a(b.seq_id[i], sizeof(llama_seq_id) * (size_t) b.n_seq_id[i], h);
        }
    }
    snprintf(buf, sizeof(buf), "%06u %s n=%d pos=%d seq=%d log=%d h=%016llx",
             op_seq, is_encode ? "encode" : "decode", b.n_tokens,
             b.pos != nullptr, b.n_seq_id != nullptr, b.logits != nullptr,
             (unsigned long long) h);
    return buf;
}

static std::string log_ctrl(const pipe_tp_ctrl & c) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%06u ctrl op=%u b0=%u a=%d b=%d c=%d d=%d",
             c.op_seq, (unsigned) c.op, (unsigned) c.b0, c.a, c.b, c.c, c.d);
    return buf;
}

// ---------------------------------------------------------------------------------------------
// the two ranks
// ---------------------------------------------------------------------------------------------

static pipe_tp_hello make_hello(uint32_t rank_first, uint32_t n_local) {
    pipe_tp_hello h = {};
    h.version = PIPE_TP_PROTO_VERSION;
    h.n_world = 2; h.rank_first = rank_first; h.n_local = n_local;
    h.n_ctx = 4096; h.n_ctx_seq = 4096; h.n_rs_seq = 4; h.n_batch = 2048; h.n_ubatch = 512;
    h.n_seq_max = 1; h.type_k = 30; h.type_v = 30;
    h.n_vocab = 248320; h.n_embd = 5120; h.n_layer = 64;
    h.flags = PIPE_TP_HELLO_F_FLASH_ATTN | PIPE_TP_HELLO_F_CAUSAL_ATTN;
    h.model_hash = 0xfeedfacecafebeefull;
    h.split_hash = 0x00c0ffeeull;
    return h;
}

static bool hello_exchange(pipe_tp_comm & c, const pipe_tp_hello & mine, std::string * err) {
    if (!c.send_msg(PIPE_TP_MSG_HELLO, &mine, sizeof(mine))) {
        *err = "send HELLO failed";
        return false;
    }
    uint8_t type = 0;
    std::vector<uint8_t> buf;
    if (!c.recv_msg(&type, buf) || type != PIPE_TP_MSG_HELLO || buf.size() != sizeof(mine)) {
        *err = "recv HELLO failed";
        return false;
    }
    pipe_tp_hello theirs;
    memcpy(&theirs, buf.data(), sizeof(theirs));
    const std::string bad = pipe_tp_hello_mismatch(mine, theirs);
    if (!bad.empty()) {
        *err = "HELLO mismatch: " + bad;
        return false;
    }
    return true;
}

struct rank_result {
    bool                     ok = false;
    std::string              err;
    std::vector<std::string> log;
    std::vector<float>       last_sum;
};

// The leader: mirrors, then "performs". Exactly the order llama_context::decode() uses.
static void run_leader(const std::vector<script_op> & script, rank_result * out,
                       int port, bool do_listen) {
    // The leader is the rank that MIRRORS. Whether it binds or dials is a separate question,
    // answered by the firewall rather than by the topology, and nothing below depends on it.
    auto comm = do_listen
        ? pipe_tp_comm::listen (TEST_HOST, port, 1 << 16, 5000)
        : pipe_tp_comm::connect(TEST_HOST, port, 1 << 16, 5000);
    if (!comm) {
        out->err = do_listen ? "leader listen failed" : "leader connect failed";
        return;
    }
    if (!hello_exchange(*comm, make_hello(0, 1), &out->err)) {
        return;
    }

    uint32_t op_seq = 0;
    std::vector<uint8_t> wire;
    std::vector<float>   partial;

    for (const auto & op : script) {
        if (op.kind == script_op::REDUCE) {
            // rank 0's partial: 1.0, 2.0, 3.0, ...
            partial.assign(op.n_values, 0.0f);
            for (size_t i = 0; i < op.n_values; i++) {
                partial[i] = (float) (i + 1);
            }
            if (!comm->exchange_add(partial.data(), partial.size(), true)) {
                out->err = "leader exchange_add failed";
                return;
            }
            out->last_sum = partial;
            continue;
        }

        if (op.kind == script_op::CTRL) {
            pipe_tp_ctrl c = op.ctrl;
            c.op_seq = ++op_seq;
            pipe_tp_encode_ctrl(c, wire);
            if (!comm->send_msg(PIPE_TP_MSG_CTRL, wire.data(), wire.size())) {
                out->err = "leader send CTRL failed";
                return;
            }
            out->log.push_back(log_ctrl(c));
            continue;
        }

        // BATCH
        const op_batch & ob = op.batch;
        std::vector<llama_seq_id *> ptrs;
        llama_batch b = {};
        b.n_tokens = (int32_t) ob.token.size();
        b.token    = const_cast<llama_token *>(ob.token.data());
        b.pos      = ob.with_pos ? const_cast<llama_pos *>(ob.pos.data()) : nullptr;
        b.logits   = ob.with_logits ? const_cast<int8_t *>(ob.logits.data()) : nullptr;
        if (ob.with_seq) {
            ptrs.assign(ob.n_seq_id.size(), nullptr);
            size_t off = 0;
            for (size_t i = 0; i < ob.n_seq_id.size(); i++) {
                ptrs[i] = const_cast<llama_seq_id *>(ob.seq_flat.data()) + off;
                off += (size_t) ob.n_seq_id[i];
            }
            b.n_seq_id = const_cast<int32_t *>(ob.n_seq_id.data());
            b.seq_id   = ptrs.data();
        }

        std::string err;
        if (!pipe_tp_encode_batch(b, ob.is_encode, op_seq + 1, wire, &err)) {
            out->err = "leader encode batch failed: " + err;
            return;
        }
        if (!comm->send_msg(PIPE_TP_MSG_DECODE, wire.data(), wire.size())) {
            out->err = "leader send DECODE failed";
            return;
        }
        op_seq++;
        out->log.push_back(log_batch(op_seq, b, ob.is_encode));
    }

    pipe_tp_ctrl bye;
    bye.op_seq = ++op_seq;
    bye.op     = PIPE_TP_CTRL_SHUTDOWN;
    pipe_tp_encode_ctrl(bye, wire);
    if (!comm->send_msg(PIPE_TP_MSG_CTRL, wire.data(), wire.size())) {
        out->err = "leader send SHUTDOWN failed";
        return;
    }
    out->ok = true;
}

// The follower: the message loop of llama_context::tp_follower_step(), with the KV replaced by a
// call log. A reduce is not a message it waits for - it is what its graph produces once it has
// been told to decode, so it runs one immediately after each applied batch, exactly as the real
// follower does.
static void run_follower(const std::vector<script_op> & script, rank_result * out,
                         int port, bool do_listen) {
    auto comm = do_listen
        ? pipe_tp_comm::listen (TEST_HOST, port, 1 << 16, 5000)
        : pipe_tp_comm::connect(TEST_HOST, port, 1 << 16, 5000);
    if (!comm) {
        out->err = do_listen ? "follower listen failed" : "follower connect failed";
        return;
    }
    if (!hello_exchange(*comm, make_hello(1, 1), &out->err)) {
        return;
    }

    // The follower knows the reduce widths only because its graph is the same graph; here that
    // knowledge comes from walking the same script, which is what "the same graph" means.
    size_t next_reduce = 0;
    auto pending_reduce = [&]() -> size_t {
        while (next_reduce < script.size() && script[next_reduce].kind != script_op::REDUCE) {
            next_reduce++;
        }
        return next_reduce < script.size() ? script[next_reduce++].n_values : 0;
    };

    uint32_t op_seq = 0;
    std::vector<uint8_t> buf;
    std::vector<float>   partial;

    for (;;) {
        uint8_t type = 0;
        if (!comm->recv_msg(&type, buf)) {
            out->err = "follower recv failed";
            return;
        }
        std::string err;

        if (type == PIPE_TP_MSG_CTRL) {
            pipe_tp_ctrl c;
            if (!pipe_tp_decode_ctrl(buf.data(), buf.size(), &c, &err)) {
                out->err = "follower bad CTRL: " + err;
                return;
            }
            if (c.op == PIPE_TP_CTRL_SHUTDOWN) {
                out->ok = true;
                return;
            }
            if (c.op_seq != op_seq + 1) {
                out->err = "follower CTRL out of lockstep";
                return;
            }
            op_seq = c.op_seq;
            out->log.push_back(log_ctrl(c));
            continue;
        }

        if (type != PIPE_TP_MSG_DECODE) {
            out->err = "follower unexpected message type";
            return;
        }
        pipe_tp_batch_desc desc;
        if (!pipe_tp_decode_batch(buf.data(), buf.size(), &desc, &err)) {
            out->err = "follower bad DECODE: " + err;
            return;
        }
        if (desc.op_seq != op_seq + 1) {
            out->err = "follower DECODE out of lockstep";
            return;
        }
        op_seq = desc.op_seq;
        llama_batch b = desc.view();
        out->log.push_back(log_batch(op_seq, b, desc.is_encode));

        // "run the graph": one reduce per mirrored batch in this script
        const size_t n = pending_reduce();
        if (n > 0) {
            // rank 1's partial: -1.0, -2.0, ... so the total is exactly zero everywhere, which
            // makes a mis-paired exchange obvious rather than merely different.
            partial.assign(n, 0.0f);
            for (size_t i = 0; i < n; i++) {
                partial[i] = -(float) (i + 1);
            }
            if (!comm->exchange_add(partial.data(), partial.size(), false)) {
                out->err = "follower exchange_add failed";
                return;
            }
            out->last_sum = partial;
        }
    }
}

// ---------------------------------------------------------------------------------------------

// `leader_listens` flips ONLY the socket direction. Rank 0 stays the leader in both arms: it is
// the one that mirrors, and it is the one that passes local_is_rank0=true to the fixed-order add.
// If any of the lockstep machinery had quietly taken "listener" to mean "rank 0", the flipped arm
// would produce a different call log or a non-cancelling reduce, and both are checked below.
static bool run_lockstep_arm(int port, bool leader_listens) {
    const std::vector<script_op> script = build_script();

    rank_result r0, r1;
    std::thread t0([&]{ run_leader  (script, &r0, port,  leader_listens); });
    std::thread t1([&]{ run_follower(script, &r1, port, !leader_listens); });
    t0.join();
    t1.join();

    if (!r0.ok || !r1.ok) {
        fprintf(stderr, "FAIL: leader ok=%d (%s), follower ok=%d (%s)\n",
                (int) r0.ok, r0.err.c_str(), (int) r1.ok, r1.err.c_str());
        return false;
    }
    if (r0.log.size() != r1.log.size()) {
        fprintf(stderr, "FAIL: leader applied %zu operations, follower %zu\n",
                r0.log.size(), r1.log.size());
        return false;
    }
    for (size_t i = 0; i < r0.log.size(); i++) {
        if (r0.log[i] != r1.log[i]) {
            fprintf(stderr, "FAIL: operation %zu differs\n  leader:   %s\n  follower: %s\n",
                    i, r0.log[i].c_str(), r1.log[i].c_str());
            return false;
        }
    }
    // Every reduce in the script paired up: rank 0's k and rank 1's -k summed to exactly zero.
    if (r0.last_sum.size() != r1.last_sum.size() || r0.last_sum.empty()) {
        fprintf(stderr, "FAIL: reduce widths did not match (%zu vs %zu)\n",
                r0.last_sum.size(), r1.last_sum.size());
        return false;
    }
    for (size_t i = 0; i < r0.last_sum.size(); i++) {
        if (r0.last_sum[i] != 0.0f || r1.last_sum[i] != 0.0f) {
            fprintf(stderr, "FAIL: reduce %zu did not cancel: %f / %f\n",
                    i, r0.last_sum[i], r1.last_sum[i]);
            return false;
        }
    }

    printf("  %zu operations applied identically on both ranks, interleaved with reduces (%s)\n",
           r0.log.size(), leader_listens ? "leader listening" : "leader dialling out");
    return true;
}

static bool test_loopback_lockstep() {
    return run_lockstep_arm(TEST_PORT, true);
}

// THE FIREWALL ARM. mad-lab-main runs ufw default-deny inbound with no allow rules, so nothing can
// dial IN to rank 0; every existing rig has main connecting OUT to 2026's already-open worker
// ports. So the follower binds and the LEADER dials - the reverse of every launch line written so
// far - and all of prebind, HELLO, the operation counter and the fixed-order add must be
// indifferent to it.
static bool test_follower_listens_leader_connects() {
    return run_lockstep_arm(TEST_PORT + 3, false);
}

// A leader that skips ONE mirror - the failure mode this whole design is built to catch, and the
// one an unaudited llama_memory_* call site would produce. The follower must refuse the very next
// frame rather than applying it at the wrong counter.
static bool test_dropped_mirror_is_caught() {
    std::atomic<bool> follower_rejected{false};
    std::string       follower_err;

    std::thread t0([&]{
        auto comm = pipe_tp_comm::listen(TEST_HOST, TEST_PORT + 1, 1024, 5000);
        if (!comm) {
            return;
        }
        std::string err;
        if (!hello_exchange(*comm, make_hello(0, 1), &err)) {
            return;
        }
        std::vector<uint8_t> wire;
        // operations 1 and 2 are sent; operation 3 is silently skipped; 4 is sent.
        for (uint32_t seq : {1u, 2u, 4u}) {
            pipe_tp_ctrl c;
            c.op_seq = seq;
            c.op     = PIPE_TP_CTRL_SEQ_RM;
            c.a = 0; c.b = 0; c.c = -1; c.d = 0;
            pipe_tp_encode_ctrl(c, wire);
            if (!comm->send_msg(PIPE_TP_MSG_CTRL, wire.data(), wire.size())) {
                return;
            }
        }
    });

    std::thread t1([&]{
        auto comm = pipe_tp_comm::connect(TEST_HOST, TEST_PORT + 1, 1024, 5000);
        if (!comm) {
            follower_err = "connect failed";
            return;
        }
        std::string err;
        if (!hello_exchange(*comm, make_hello(1, 1), &err)) {
            follower_err = err;
            return;
        }
        uint32_t op_seq = 0;
        std::vector<uint8_t> buf;
        for (int i = 0; i < 3; i++) {
            uint8_t type = 0;
            if (!comm->recv_msg(&type, buf)) {
                break;
            }
            pipe_tp_ctrl c;
            std::string e;
            if (!pipe_tp_decode_ctrl(buf.data(), buf.size(), &c, &e)) {
                break;
            }
            if (c.op_seq != op_seq + 1) {
                follower_rejected = true;
                return;
            }
            op_seq = c.op_seq;
        }
    });

    t0.join();
    t1.join();

    if (!follower_rejected) {
        fprintf(stderr, "FAIL: the follower accepted a gap in the operation counter (%s)\n",
                follower_err.c_str());
        return false;
    }
    printf("  a dropped mirror is rejected at the very next frame\n");
    return true;
}


// The first two-machine run died here, not in any of the logic above: rank 0 only bound its peer
// port after its model load, and the two loads differ by minutes (~7 min off a spinning disk on
// the leader against ~43 s on the follower), so the follower's connect window opened and closed
// against a port that did not exist yet.
//
// The fix is prebind(): rank 0 opens the listening socket before it touches a weight. This test
// pins the property the fix depends on, which is not obvious and is easy to regress - once
// listen() has been called the KERNEL completes the peer's TCP handshake from the backlog, so the
// follower's connect() returns IMMEDIATELY even though the leader will not call accept() for
// another second and a half, and the follower can send its HELLO into a socket nobody is reading
// yet. If prebind ever stopped actually listening, the connect below would fall into the retry
// path and take at least as long as the leader's simulated load, which is what this asserts
// against.
static bool test_prebind_lets_the_follower_connect_during_the_load() {
    const int port = TEST_PORT + 2;

    if (!pipe_tp_comm::prebind(TEST_HOST, port)) {
        fprintf(stderr, "FAIL: prebind(%s:%d) failed\n", TEST_HOST, port);
        return false;
    }

    std::atomic<long long> connect_ms{-1};
    std::atomic<bool>      follower_ok{false};
    std::atomic<bool>      leader_ok{false};

    std::thread t1([&]{
        const auto t0 = std::chrono::steady_clock::now();
        // A single attempt: no retry loop is allowed to rescue this. If the port is not already
        // listening, this fails and the test fails with it.
        auto comm = pipe_tp_comm::connect(TEST_HOST, port, 1024, 0);
        connect_ms = (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (!comm) {
            return;
        }
        // Send HELLO into a socket the leader has not accepted yet - it sits in the kernel's
        // receive buffer until the leader gets there, which is what lets the follower stop caring
        // how long the leader's load takes.
        std::string err;
        if (!hello_exchange(*comm, make_hello(1, 1), &err)) {
            fprintf(stderr, "  follower hello: %s\n", err.c_str());
            return;
        }
        follower_ok = true;
    });

    std::thread t0([&]{
        // "loading the model"
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        auto comm = pipe_tp_comm::listen(TEST_HOST, port, 1024, 30000);
        if (!comm) {
            return;
        }
        std::string err;
        if (!hello_exchange(*comm, make_hello(0, 1), &err)) {
            fprintf(stderr, "  leader hello: %s\n", err.c_str());
            return;
        }
        leader_ok = true;
    });

    t0.join();
    t1.join();

    if (connect_ms < 0) {
        fprintf(stderr, "FAIL: the follower could not connect to a prebound port at all\n");
        return false;
    }
    if (connect_ms > 500) {
        fprintf(stderr, "FAIL: connect to a prebound port took %lld ms - it should complete from "
                        "the listen backlog immediately, not wait for the leader's accept\n",
                (long long) connect_ms);
        return false;
    }
    if (!follower_ok || !leader_ok) {
        fprintf(stderr, "FAIL: handshake across the load window failed (follower ok=%d, leader ok=%d)\n",
                (int) follower_ok, (int) leader_ok);
        return false;
    }
    printf("  follower connected in %lld ms and its HELLO waited out a 1500 ms leader load\n",
           (long long) connect_ms);
    return true;
}

// The auto rule that keeps --tp-listen optional: an address this machine can bind is a "listen
// here", one it cannot is a "dial there". Every launch line written before the firewall problem
// gave rank 0 a wildcard or loopback address, so all of them still resolve to "rank 0 listens".
static bool test_host_is_local() {
    struct { const char * host; bool want; } cases[] = {
        {"0.0.0.0",       true },  // the wildcard: what every rank-0 launch line has used
        {"127.0.0.1",     true },  // the G1 loopback tripwire
        {"127.1.2.3",     true },
        {"192.168.1.33",  false},  // rank 1's address, seen from rank 0 -> dial it
        {"203.0.113.7",   false},  // TEST-NET-3, cannot be assigned here
        {"mad-lab-main",  false},  // a NAME: create_server cannot bind one at all
        {"",              false},
    };
    for (auto & c : cases) {
        const bool got = pipe_tp_comm::host_is_local(c.host);
        if (got != c.want) {
            fprintf(stderr, "FAIL: host_is_local(\"%s\") = %d, expected %d\n", c.host, (int) got, (int) c.want);
            return false;
        }
    }
    printf("  the wildcard and loopback bind here; a peer address and a name do not\n");
    return true;
}

int main() {
    if (const char * p = getenv("WP_TP_TEST_PORT")) {
        TEST_PORT = atoi(p);
    }
    struct { const char * name; bool (*fn)(); } tests[] = {
        {"loopback-lockstep",  test_loopback_lockstep},
        {"follower-listens",   test_follower_listens_leader_connects},
        {"dropped-mirror",     test_dropped_mirror_is_caught},
        {"prebind-ordering",   test_prebind_lets_the_follower_connect_during_the_load},
        {"host-is-local",      test_host_is_local},
    };
    for (auto & t : tests) {
        printf("test-tp-lockstep: %s\n", t.name);
        if (!t.fn()) {
            fprintf(stderr, "test-tp-lockstep: %s FAILED\n", t.name);
            return 1;
        }
    }
    printf("test-tp-lockstep: OK\n");
    return 0;
}
