// Cross-host tensor parallelism, M3: the batch-descriptor and control-message codec.
//
// This is the half of the follower that can be tested with no model, no GPU and no socket: given
// a llama_batch, the bytes that go on the wire must decode back into a batch that llama_decode
// cannot tell apart from the original - same tokens, same positions, same per-token sequence sets,
// same logits flags, AND the same NULLNESS of each optional array, because llama_batch_allocr
// treats a null pos/seq_id/logits differently from a filled-in one and the two ranks must make the
// same choice.
//
// The two shapes that matter in a real run are covered explicitly: a 512-token prefill ubatch (the
// widest thing the rig sends) and a 1-token decode step (the thing it sends thousands of times).

#include "pipe-tp-msg.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_checks = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        g_checks++;                                                                    \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
            return false;                                                              \
        }                                                                              \
    } while (0)

// A llama_batch plus the storage it points into, so a test batch can be built and outlive the
// expression that made it.
struct test_batch {
    std::vector<llama_token>   token;
    std::vector<llama_pos>     pos;
    std::vector<int32_t>       n_seq_id;
    std::vector<llama_seq_id>  seq_flat;
    std::vector<llama_seq_id*> seq_ptr;
    std::vector<int8_t>        logits;

    llama_batch b = {};

    void finish(bool with_pos, bool with_seq, bool with_logits) {
        b = llama_batch{};
        b.n_tokens = (int32_t) token.size();
        b.token    = token.data();
        b.embd     = nullptr;
        b.pos      = with_pos ? pos.data() : nullptr;
        b.logits   = with_logits ? logits.data() : nullptr;
        if (with_seq) {
            seq_ptr.assign(n_seq_id.size(), nullptr);
            size_t off = 0;
            for (size_t i = 0; i < n_seq_id.size(); i++) {
                seq_ptr[i] = seq_flat.data() + off;
                off += (size_t) n_seq_id[i];
            }
            b.n_seq_id = n_seq_id.data();
            b.seq_id   = seq_ptr.data();
        } else {
            b.n_seq_id = nullptr;
            b.seq_id   = nullptr;
        }
    }
};

// Compare a decoded descriptor's view against the batch it came from, field by field, including
// which optional arrays are null.
static bool batches_match(const llama_batch & a, const llama_batch & b) {
    CHECK(a.n_tokens == b.n_tokens);
    CHECK((a.pos      == nullptr) == (b.pos      == nullptr));
    CHECK((a.n_seq_id == nullptr) == (b.n_seq_id == nullptr));
    CHECK((a.seq_id   == nullptr) == (b.seq_id   == nullptr));
    CHECK((a.logits   == nullptr) == (b.logits   == nullptr));
    CHECK(a.embd == nullptr && b.embd == nullptr);

    for (int32_t i = 0; i < a.n_tokens; i++) {
        CHECK(a.token[i] == b.token[i]);
        if (a.pos)    CHECK(a.pos[i]    == b.pos[i]);
        if (a.logits) CHECK(a.logits[i] == b.logits[i]);
        if (a.n_seq_id) {
            CHECK(a.n_seq_id[i] == b.n_seq_id[i]);
            for (int32_t j = 0; j < a.n_seq_id[i]; j++) {
                CHECK(a.seq_id[i][j] == b.seq_id[i][j]);
            }
        }
    }
    return true;
}

static bool round_trip(test_batch & tb, bool is_encode, uint32_t op_seq) {
    std::vector<uint8_t> wire;
    std::string err;
    CHECK(pipe_tp_encode_batch(tb.b, is_encode, op_seq, wire, &err));
    CHECK(err.empty());

    pipe_tp_batch_desc desc;
    CHECK(pipe_tp_decode_batch(wire.data(), wire.size(), &desc, &err));
    CHECK(err.empty());
    CHECK(desc.op_seq == op_seq);
    CHECK(desc.is_encode == is_encode);

    llama_batch got = desc.view();
    CHECK(batches_match(tb.b, got));

    // view() is called more than once in the follower's lifetime (once per message); the rebuilt
    // seq_id pointer table must stay correct.
    llama_batch again = desc.view();
    CHECK(batches_match(tb.b, again));
    return true;
}

// ---------------------------------------------------------------------------------------------

static bool test_prefill_512() {
    // A 512-token prefill ubatch, one sequence, only the last token producing logits: exactly the
    // shape tools/server sends for the first chunk of a prompt.
    test_batch tb;
    for (int i = 0; i < 512; i++) {
        tb.token.push_back(1000 + i * 7);
        tb.pos.push_back(i);
        tb.n_seq_id.push_back(1);
        tb.seq_flat.push_back(0);
        tb.logits.push_back(i == 511 ? 1 : 0);
    }
    tb.finish(true, true, true);
    CHECK(round_trip(tb, false, 1));

    // Size sanity: the descriptor must be a few KB, not a fraction of the model. 512 tokens x
    // (token + pos + n_seq_id + seq_id + logits) is ~8.7 KB; the point of mirroring the BATCH and
    // not the ubatches is that this is paid once per llama_decode, not once per graph.
    std::vector<uint8_t> wire;
    std::string err;
    CHECK(pipe_tp_encode_batch(tb.b, false, 1, wire, &err));
    CHECK(wire.size() > 512 * 4 && wire.size() < 32 * 1024);
    printf("  512-token prefill descriptor: %zu bytes\n", wire.size());
    return true;
}

static bool test_decode_1() {
    // The single-token decode step, thousands of times per request.
    test_batch tb;
    tb.token.push_back(151643);
    tb.pos.push_back(4095);
    tb.n_seq_id.push_back(1);
    tb.seq_flat.push_back(0);
    tb.logits.push_back(1);
    tb.finish(true, true, true);
    CHECK(round_trip(tb, false, 2));

    std::vector<uint8_t> wire;
    std::string err;
    CHECK(pipe_tp_encode_batch(tb.b, false, 2, wire, &err));
    printf("  1-token decode descriptor:    %zu bytes\n", wire.size());
    CHECK(wire.size() < 64);
    return true;
}

static bool test_optional_arrays_null() {
    // llama_batch_get_one() - which is what the warm-up run and llama-cli use - leaves pos,
    // n_seq_id, seq_id and logits ALL null and lets llama_batch_allocr fill in the defaults. If
    // the mirror turned those into explicit arrays, the follower's allocator would take a
    // different branch than the leader's. Nullness is part of the payload.
    test_batch tb;
    tb.token = {1, 2, 3};
    tb.finish(false, false, false);
    CHECK(tb.b.pos == nullptr && tb.b.n_seq_id == nullptr && tb.b.logits == nullptr);
    CHECK(round_trip(tb, false, 3));

    // ... and every partial combination.
    for (int mask = 0; mask < 8; mask++) {
        test_batch t2;
        for (int i = 0; i < 5; i++) {
            t2.token.push_back(i);
            t2.pos.push_back(100 + i);
            t2.n_seq_id.push_back(1);
            t2.seq_flat.push_back(0);
            t2.logits.push_back(i % 2);
        }
        t2.finish(mask & 1, mask & 2, mask & 4);
        CHECK(round_trip(t2, false, 4 + mask));
    }
    return true;
}

static bool test_multi_seq() {
    // Ragged per-token sequence sets, including a token that belongs to no sequence at all. This
    // is the flattening the wire format has to get right; an off-by-one here would silently
    // attribute a token to the wrong slot on rank 1.
    test_batch tb;
    const int32_t counts[] = {1, 3, 0, 2, 1};
    llama_seq_id next = 0;
    for (int i = 0; i < 5; i++) {
        tb.token.push_back(500 + i);
        tb.pos.push_back(i);
        tb.n_seq_id.push_back(counts[i]);
        for (int32_t j = 0; j < counts[i]; j++) {
            tb.seq_flat.push_back(next++ % 4);
        }
        tb.logits.push_back(0);
    }
    tb.finish(true, true, true);
    CHECK(round_trip(tb, false, 20));
    CHECK(round_trip(tb, true,  21)); // the encode() path uses the same descriptor
    return true;
}

static bool test_rejections() {
    std::vector<uint8_t> wire;
    std::string err;

    // An embd batch is refused rather than half-supported: spec D.2/R7 - the follower is a token
    // path, and a hidden-state path would also need n_pos_per_embd positions per token.
    float dummy = 0.0f;
    llama_batch b = {};
    b.n_tokens = 1;
    b.embd     = &dummy;
    CHECK(!pipe_tp_encode_batch(b, false, 1, wire, &err));
    CHECK(err.find("embd") != std::string::npos);

    // No tokens.
    llama_token tok = 1;
    b = llama_batch{};
    b.n_tokens = 0;
    b.token    = &tok;
    CHECK(!pipe_tp_encode_batch(b, false, 1, wire, &err));

    // n_seq_id without seq_id (or the reverse) is a malformed batch, not a defaultable one.
    int32_t nsi = 1;
    b = llama_batch{};
    b.n_tokens = 1;
    b.token    = &tok;
    b.n_seq_id = &nsi;
    b.seq_id   = nullptr;
    CHECK(!pipe_tp_encode_batch(b, false, 1, wire, &err));
    return true;
}

static bool test_corruption_is_caught() {
    // Every frame carries an FNV-1a of its body. A single flipped bit anywhere - a truncated read,
    // a reordered frame, a wrong-length descriptor - has to be loud, because the alternative is
    // two ranks quietly decoding different batches and producing a plausible wrong answer.
    test_batch tb;
    for (int i = 0; i < 40; i++) {
        tb.token.push_back(i);
        tb.pos.push_back(i);
        tb.n_seq_id.push_back(1);
        tb.seq_flat.push_back(0);
        tb.logits.push_back(0);
    }
    tb.finish(true, true, true);

    std::vector<uint8_t> wire;
    std::string err;
    CHECK(pipe_tp_encode_batch(tb.b, false, 9, wire, &err));

    for (size_t i = 0; i < wire.size(); i++) {
        std::vector<uint8_t> bad = wire;
        bad[i] ^= 0x01;
        pipe_tp_batch_desc desc;
        std::string e2;
        CHECK(!pipe_tp_decode_batch(bad.data(), bad.size(), &desc, &e2));
    }
    // ... and truncation at every length.
    for (size_t n = 0; n < wire.size(); n++) {
        pipe_tp_batch_desc desc;
        std::string e2;
        CHECK(!pipe_tp_decode_batch(wire.data(), n, &desc, &e2));
    }
    return true;
}

static bool test_ctrl_round_trip() {
    const uint8_t ops[] = {
        PIPE_TP_CTRL_MEM_CLEAR, PIPE_TP_CTRL_SEQ_RM, PIPE_TP_CTRL_SEQ_CP,
        PIPE_TP_CTRL_SEQ_KEEP,  PIPE_TP_CTRL_SEQ_ADD, PIPE_TP_CTRL_SEQ_DIV,
        PIPE_TP_CTRL_SHUTDOWN,
    };
    uint32_t seq = 0;
    for (uint8_t op : ops) {
        pipe_tp_ctrl in;
        in.op_seq = ++seq;
        in.op     = op;
        in.b0     = (uint8_t) (op % 2);
        // Negative positions are the norm here: seq_rm(seq, p0, -1) means "to the end".
        in.a = 3; in.b = -1; in.c = 4096; in.d = -1024;

        std::vector<uint8_t> wire;
        pipe_tp_encode_ctrl(in, wire);
        CHECK(wire.size() == 32); // fixed size: a control frame must not depend on its arguments

        pipe_tp_ctrl out;
        std::string err;
        CHECK(pipe_tp_decode_ctrl(wire.data(), wire.size(), &out, &err));
        CHECK(out.op_seq == in.op_seq && out.op == in.op && out.b0 == in.b0);
        CHECK(out.a == in.a && out.b == in.b && out.c == in.c && out.d == in.d);

        for (size_t i = 0; i < wire.size(); i++) {
            std::vector<uint8_t> bad = wire;
            bad[i] ^= 0x80;
            pipe_tp_ctrl o2;
            std::string e2;
            CHECK(!pipe_tp_decode_ctrl(bad.data(), bad.size(), &o2, &e2));
        }
    }
    return true;
}

static pipe_tp_hello base_hello() {
    pipe_tp_hello h = {};
    h.version = PIPE_TP_PROTO_VERSION;
    h.n_world = 4; h.rank_first = 0; h.n_local = 2;
    h.n_ctx = 4096; h.n_ctx_seq = 4096; h.n_rs_seq = 4; h.n_batch = 2048; h.n_ubatch = 512;
    h.n_seq_max = 1;
    h.type_k = 30; h.type_v = 30;
    h.n_vocab = 248320; h.n_embd = 5120; h.n_layer = 64;
    h.flags = PIPE_TP_HELLO_F_FLASH_ATTN | PIPE_TP_HELLO_F_CAUSAL_ATTN;
    h.model_hash = 0x1234567890abcdefull;
    h.split_hash = 0x0fedcba098765432ull;
    return h;
}

static bool test_hello_gate() {
    pipe_tp_hello r0 = base_hello();
    pipe_tp_hello r1 = base_hello();
    r1.rank_first = 2;
    r1.n_local    = 2;

    // The good case, checked from BOTH sides: the function is called on each rank with its own
    // HELLO first, so it must be symmetric.
    CHECK(pipe_tp_hello_mismatch(r0, r1).empty());
    CHECK(pipe_tp_hello_mismatch(r1, r0).empty());

    // Every equality field must be caught. These are exactly the mistakes that produce a wrong
    // answer rather than a crash - two ranks with different -ts compute different row maps and
    // then add unrelated rows together at every reduce.
    struct { const char * what; void (*mut)(pipe_tp_hello &); } cases[] = {
        {"-ts",       [](pipe_tp_hello & h){ h.split_hash ^= 1; }},
        {"model",     [](pipe_tp_hello & h){ h.model_hash ^= 1; }},
        {"n_ctx",     [](pipe_tp_hello & h){ h.n_ctx = 8192; }},
        {"n_ubatch",  [](pipe_tp_hello & h){ h.n_ubatch = 256; }},
        {"n_batch",   [](pipe_tp_hello & h){ h.n_batch = 512; }},
        {"n_seq_max", [](pipe_tp_hello & h){ h.n_seq_max = 4; }},
        {"type_k",    [](pipe_tp_hello & h){ h.type_k = 1; }},
        {"type_v",    [](pipe_tp_hello & h){ h.type_v = 1; }},
        {"n_world",   [](pipe_tp_hello & h){ h.n_world = 2; }},
        {"flags",     [](pipe_tp_hello & h){ h.flags ^= PIPE_TP_HELLO_F_FLASH_ATTN; }},
        {"n_rs_seq",  [](pipe_tp_hello & h){ h.n_rs_seq = 0; }},
        {"n_vocab",   [](pipe_tp_hello & h){ h.n_vocab += 1; }},
        {"version",   [](pipe_tp_hello & h){ h.version += 1; }},
    };
    for (auto & c : cases) {
        pipe_tp_hello bad = r1;
        c.mut(bad);
        const std::string msg = pipe_tp_hello_mismatch(r0, bad);
        if (msg.empty()) {
            fprintf(stderr, "FAIL: a differing %s was accepted by the HELLO gate\n", c.what);
            g_checks++;
            return false;
        }
        g_checks++;
    }

    // The windows must TILE the world - the one thing the ranks must NOT agree about.
    pipe_tp_hello same = r0;                 // both claiming device 0
    CHECK(!pipe_tp_hello_mismatch(r0, same).empty());

    pipe_tp_hello gap = r1; gap.rank_first = 3; gap.n_local = 1;   // leaves device 2 unowned
    CHECK(!pipe_tp_hello_mismatch(r0, gap).empty());

    pipe_tp_hello over = r1; over.rank_first = 1;                  // overlaps device 1
    CHECK(!pipe_tp_hello_mismatch(r0, over).empty());

    pipe_tp_hello shortfall = r1; shortfall.n_local = 1;           // world ends at 3, not 4
    CHECK(!pipe_tp_hello_mismatch(r0, shortfall).empty());

    // The loopback tripwire shape (spec G1): 2 ranks x 1 device.
    pipe_tp_hello a = base_hello(); a.n_world = 2; a.rank_first = 0; a.n_local = 1;
    pipe_tp_hello b = a;            b.rank_first = 1;
    CHECK(pipe_tp_hello_mismatch(a, b).empty());
    CHECK(pipe_tp_hello_mismatch(b, a).empty());
    return true;
}

int main() {
    struct { const char * name; bool (*fn)(); } tests[] = {
        {"prefill-512",        test_prefill_512},
        {"decode-1",           test_decode_1},
        {"optional-arrays",    test_optional_arrays_null},
        {"multi-seq",          test_multi_seq},
        {"rejections",         test_rejections},
        {"corruption",         test_corruption_is_caught},
        {"ctrl-round-trip",    test_ctrl_round_trip},
        {"hello-gate",         test_hello_gate},
    };
    for (auto & t : tests) {
        printf("test-tp-msg: %s\n", t.name);
        if (!t.fn()) {
            fprintf(stderr, "test-tp-msg: %s FAILED\n", t.name);
            return 1;
        }
    }
    printf("test-tp-msg: OK (%d checks)\n", g_checks);
    return 0;
}
