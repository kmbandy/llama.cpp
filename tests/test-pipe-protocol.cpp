// Unit tests for the Phase 2 pipeline protocol framing (src/pipeline/pipe-protocol.*).
//
// No model or inference: buffer encode/decode round-trips, malformed-frame
// rejection, the HELLO mismatch matrix, and a closed-peer loopback send.
// Endianness is exercised by decoding hand-built little-endian byte strings.
//
// Standalone build:
//   g++ -std=c++17 -I include -I ggml/include -I src -I src/pipeline
//       tests/test-pipe-protocol.cpp src/pipeline/pipe-protocol.cpp
//       src/pipeline/pipe-transport.cpp src/llama-pipeline.cpp -o /tmp/t

#include "pipe-protocol.h"
#include "pipe-transport.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifndef _WIN32
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <signal.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

static int g_failed = 0;

static_assert(PIPE_HELLO == 1 && PIPE_FWD_REQ == 2 && PIPE_FWD_RESP == 3 &&
              PIPE_TOKEN == 4 && PIPE_ERROR == 5 && PIPE_PING == 6 &&
              PIPE_PONG == 7, "stage frame values changed");
static_assert(PIPE_ERR_GENERIC == 0 && PIPE_ERR_HELLO == 1 &&
              PIPE_ERR_BAD_FRAME == 2 && PIPE_ERR_STALE_SEQ == 3 &&
              PIPE_ERR_DECODE == 4, "stage error values changed");
static_assert(pipe_hello::WIRE_SIZE == 28, "stage HELLO wire size changed");

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                         \
        }                                                                       \
    } while (0)

#define CHECK_THROWS_PROTO(expr, want_code)                                     \
    do {                                                                        \
        bool threw = false;                                                     \
        try { expr; }                                                           \
        catch (const pipe_protocol_error & e) {                                 \
            threw = true;                                                       \
            if (e.code != (want_code)) {                                        \
                std::fprintf(stderr,                                            \
                    "FAIL %s:%d: %s threw code %d, want %d\n",                  \
                    __FILE__, __LINE__, #expr, (int) e.code, (int) (want_code)); \
                ++g_failed;                                                     \
            }                                                                   \
        }                                                                       \
        if (!threw) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: expected throw: %s\n",            \
                         __FILE__, __LINE__, #expr);                            \
            ++g_failed;                                                         \
        }                                                                       \
    } while (0)

// little-endian builders for hand-crafted byte strings
static void put_u32(std::vector<uint8_t> & v, uint32_t x) {
    v.push_back((uint8_t) (x >>  0));
    v.push_back((uint8_t) (x >>  8));
    v.push_back((uint8_t) (x >> 16));
    v.push_back((uint8_t) (x >> 24));
}
static void put_i32(std::vector<uint8_t> & v, int32_t x) { put_u32(v, (uint32_t) x); }

// ---------------------------------------------------------------------------
// header round-trip and rejection

static void test_header_roundtrip() {
    pipe_frame_header h;
    h.magic   = PIPE_MAGIC;
    h.version = PIPE_VERSION;
    h.type    = (uint32_t) PIPE_FWD_REQ;
    h.flags   = 0;
    h.seq_id  = 0x0102030405060708ull;
    h.length  = 12345;

    uint8_t buf[PIPE_HEADER_SIZE];
    pipe_encode_header(buf, h);

    // magic is little-endian "LLPP": 0x50 0x50 0x4C 0x4C on the wire
    CHECK(buf[0] == 0x50 && buf[1] == 0x50 && buf[2] == 0x4C && buf[3] == 0x4C);

    pipe_frame_header d = pipe_decode_header(buf);
    CHECK(d.magic   == h.magic);
    CHECK(d.version == h.version);
    CHECK(d.type    == h.type);
    CHECK(d.flags   == h.flags);
    CHECK(d.seq_id  == h.seq_id);
    CHECK(d.length  == h.length);
}

static void test_header_rejects() {
    uint8_t buf[PIPE_HEADER_SIZE];

    // bad magic
    {
        pipe_frame_header h{0xDEADBEEF, PIPE_VERSION, 1, 0, 0, 0};
        pipe_encode_header(buf, h);
        CHECK_THROWS_PROTO(pipe_decode_header(buf), PIPE_ERR_BAD_FRAME);
    }
    // wrong version
    {
        pipe_frame_header h{PIPE_MAGIC, 2, 1, 0, 0, 0};
        pipe_encode_header(buf, h);
        CHECK_THROWS_PROTO(pipe_decode_header(buf), PIPE_ERR_BAD_FRAME);
    }
    // oversized length: must be rejected, never honored with an allocation
    {
        pipe_frame_header h{PIPE_MAGIC, PIPE_VERSION, 2, 0, 0, 4ull * 1024 * 1024 * 1024};
        pipe_encode_header(buf, h);
        CHECK_THROWS_PROTO(pipe_decode_header(buf), PIPE_ERR_BAD_FRAME);
    }
    // exactly PIPE_MAX_PAYLOAD is allowed (the boundary is the payload alloc,
    // which the caller controls)
    {
        pipe_frame_header h{PIPE_MAGIC, PIPE_VERSION, 2, 0, 0, PIPE_MAX_PAYLOAD};
        pipe_encode_header(buf, h);
        pipe_frame_header d = pipe_decode_header(buf);
        CHECK(d.length == PIPE_MAX_PAYLOAD);
    }
    // truncated header: only testable via the socket path, but decode of a
    // short logical frame is covered by the payload decoders below
}

// ---------------------------------------------------------------------------
// HELLO round-trip

static void test_hello_roundtrip() {
    pipe_hello h;
    h.role        = PIPE_ROLE_HEAD;
    h.layer_first = 0;
    h.layer_last  = 56;
    h.n_layer     = 78;
    h.n_embd      = 6144;
    h.hidden_type = PIPE_HIDDEN_F32;
    h.model_hash  = 0xA5A5A5A5u;

    std::vector<uint8_t> enc = pipe_encode_hello(h);
    CHECK(enc.size() == pipe_hello::WIRE_SIZE);

    pipe_hello d = pipe_decode_hello(enc.data(), enc.size());
    CHECK(d.role        == h.role);
    CHECK(d.layer_first == h.layer_first);
    CHECK(d.layer_last  == h.layer_last);
    CHECK(d.n_layer     == h.n_layer);
    CHECK(d.n_embd      == h.n_embd);
    CHECK(d.hidden_type == h.hidden_type);
    CHECK(d.model_hash  == h.model_hash);

    // wrong size
    std::vector<uint8_t> short_buf(enc.begin(), enc.end() - 1);
    CHECK_THROWS_PROTO(pipe_decode_hello(short_buf.data(), short_buf.size()), PIPE_ERR_BAD_FRAME);
    // bad role
    std::vector<uint8_t> bad = enc;
    bad[0] = 9; bad[1] = 0; bad[2] = 0; bad[3] = 0;
    CHECK_THROWS_PROTO(pipe_decode_hello(bad.data(), bad.size()), PIPE_ERR_HELLO);
}

static void test_expert_hello_roundtrip() {
    pipe_expert_hello h;
    h.role           = PIPE_EXPERT_ROLE_WORKER;
    h.hidden_type    = PIPE_HIDDEN_F16;
    h.n_embd         = 6144;
    h.n_ff_exp       = 2048;
    h.n_expert       = 256;
    h.n_expert_used  = 8;
    h.expert_first   = 85;
    h.expert_last    = 255;
    h.n_slots        = 4;
    h.layers         = { 3, 4 };
    h.model_identity = "sha256:logical-model";
    h.shard_identity = "sha256:shard-85-255";

    const std::vector<uint8_t> enc = pipe_encode_expert_hello(h);
    const pipe_expert_hello d = pipe_decode_expert_hello(enc.data(), enc.size());
    CHECK(d.role           == h.role);
    CHECK(d.layers         == h.layers);
    CHECK(d.model_identity == h.model_identity);
    CHECK(d.shard_identity == h.shard_identity);

    const pipe_expert_hello_ack accepted{ true, "" };
    const std::vector<uint8_t> accepted_enc =
        pipe_encode_expert_hello_ack(accepted);
    const pipe_expert_hello_ack accepted_dec =
        pipe_decode_expert_hello_ack(accepted_enc.data(), accepted_enc.size());
    CHECK(accepted_dec.accepted);
    CHECK(accepted_dec.reason.empty());

    const pipe_expert_hello_ack rejected{ false, "logical model mismatch" };
    const std::vector<uint8_t> rejected_enc =
        pipe_encode_expert_hello_ack(rejected);
    const pipe_expert_hello_ack rejected_dec =
        pipe_decode_expert_hello_ack(rejected_enc.data(), rejected_enc.size());
    CHECK(!rejected_dec.accepted);
    CHECK(rejected_dec.reason == rejected.reason);
}

#ifndef _WIN32
static int reserve_transport_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(fd >= 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    CHECK(bind(fd, (sockaddr *) &address, sizeof(address)) == 0);
    socklen_t length = sizeof(address);
    CHECK(getsockname(fd, (sockaddr *) &address, &length) == 0);
    const int port = ntohs(address.sin_port);
    close(fd);
    return port;
}

static void test_closed_peer_is_transport_error() {
    const int port = reserve_transport_port();
    int ready[2];
    CHECK(port > 0);
    CHECK(pipe(ready) == 0);
    const pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        close(ready[0]);
        signal(SIGPIPE, SIG_DFL);
        pipe_socket_ptr server =
            pipe_socket_t::create_server("127.0.0.1", port);
        if (!server) {
            _exit(2);
        }
        const uint8_t marker = 1;
        if (write(ready[1], &marker, 1) != 1) {
            _exit(3);
        }
        close(ready[1]);
        pipe_socket_ptr peer = server->accept();
        if (!peer) {
            _exit(4);
        }
        uint8_t byte = 0;
        if (peer->recv_data(&byte, 1)) {
            _exit(5);
        }
        for (int attempt = 0; attempt < 4; ++attempt) {
            if (!peer->send_data(&byte, 1)) {
                _exit(0);
            }
            usleep(10000);
        }
        _exit(6);
    }
    if (child < 0) {
        close(ready[0]);
        close(ready[1]);
        return;
    }

    close(ready[1]);
    uint8_t marker = 0;
    CHECK(read(ready[0], &marker, 1) == 1);
    close(ready[0]);

    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(fd >= 0);
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = htons(port);
    CHECK(connect(fd, (sockaddr *) &address, sizeof(address)) == 0);
    shutdown(fd, SHUT_RDWR);
    close(fd);

    int status = 0;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status));
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
#endif

// ---------------------------------------------------------------------------
// FWD_REQ round-trip incl. empty and large payloads

static void test_fwd_req_roundtrip() {
    const int32_t n_embd = 8;
    pipe_fwd_req r;
    r.n_tokens       = 3;
    r.n_pos_per_embd = 1;
    r.pos            = {10, 11, 12};
    r.seq_tokens     = {3};
    r.hidden.resize((size_t) n_embd * 3 * 4);
    for (size_t i = 0; i < r.hidden.size(); ++i) {
        r.hidden[i] = (uint8_t) (i & 0xff);
    }

    std::vector<uint8_t> enc = pipe_encode_fwd_req(r, PIPE_HIDDEN_F32);
    pipe_fwd_req d = pipe_decode_fwd_req(enc.data(), enc.size(), n_embd, PIPE_HIDDEN_F32);
    CHECK(d.n_tokens       == r.n_tokens);
    CHECK(d.n_pos_per_embd == r.n_pos_per_embd);
    CHECK(d.pos            == r.pos);
    CHECK(d.seq_tokens     == r.seq_tokens);
    CHECK(d.hidden         == r.hidden);

    // hidden byte count must match exactly; a truncated hidden is an error
    std::vector<uint8_t> trunc(enc.begin(), enc.end() - 1);
    CHECK_THROWS_PROTO(pipe_decode_fwd_req(trunc.data(), trunc.size(), n_embd, PIPE_HIDDEN_F32),
                       PIPE_ERR_BAD_FRAME);
}

static void test_fwd_req_empty_and_max() {
    const int32_t n_embd = 4;

    // empty (n_tokens = 0): legal shape, hidden is empty
    {
        pipe_fwd_req r;
        r.n_tokens       = 0;
        r.n_pos_per_embd = 1;
        r.seq_tokens     = {}; // n_seqs = 0: the sum check is skipped
        std::vector<uint8_t> enc = pipe_encode_fwd_req(r, PIPE_HIDDEN_F32);
        pipe_fwd_req d = pipe_decode_fwd_req(enc.data(), enc.size(), n_embd, PIPE_HIDDEN_F32);
        CHECK(d.n_tokens == 0);
        CHECK(d.hidden.empty());
    }

    // large-but-legal payload
    {
        pipe_fwd_req r;
        r.n_tokens       = 1024;
        r.n_pos_per_embd = 1;
        r.pos.assign(1024, 7);
        r.seq_tokens     = {1024};
        r.hidden.assign((size_t) n_embd * 1024 * 4, 0xAB);
        std::vector<uint8_t> enc = pipe_encode_fwd_req(r, PIPE_HIDDEN_F32);
        pipe_fwd_req d = pipe_decode_fwd_req(enc.data(), enc.size(), n_embd, PIPE_HIDDEN_F32);
        CHECK(d.hidden.size() == r.hidden.size());
        CHECK(d.pos.size() == 1024);
    }

    // seq_tokens that do not sum to n_tokens -> reject
    {
        pipe_fwd_req r;
        r.n_tokens       = 4;
        r.n_pos_per_embd = 1;
        r.pos            = {0, 1, 2, 3};
        r.seq_tokens     = {2, 1}; // sums to 3 != 4
        r.hidden.assign((size_t) n_embd * 4 * 4, 0);
        std::vector<uint8_t> enc = pipe_encode_fwd_req(r, PIPE_HIDDEN_F32);
        CHECK_THROWS_PROTO(pipe_decode_fwd_req(enc.data(), enc.size(), n_embd, PIPE_HIDDEN_F32),
                           PIPE_ERR_BAD_FRAME);
    }

    // n_pos_per_embd == 0 -> reject
    {
        std::vector<uint8_t> buf;
        put_u32(buf, 4); // n_tokens
        put_u32(buf, 0); // n_pos_per_embd = 0
        put_i32(buf, 0); put_i32(buf, 0); put_i32(buf, 0); put_i32(buf, 0);
        put_u32(buf, 1); // n_seqs
        put_i32(buf, 4);
        for (int i = 0; i < n_embd * 4 * 4; ++i) buf.push_back(0);
        CHECK_THROWS_PROTO(pipe_decode_fwd_req(buf.data(), buf.size(), n_embd, PIPE_HIDDEN_F32),
                           PIPE_ERR_BAD_FRAME);
    }
}

// ---------------------------------------------------------------------------
// FWD_RESP / TOKEN / ERROR round-trips

static void test_fwd_resp_roundtrip() {
    const int32_t n_embd = 4;
    pipe_fwd_resp r;
    r.n_tokens = 2;
    r.hidden.assign((size_t) n_embd * 2 * 4, 0x5A);
    std::vector<uint8_t> enc = pipe_encode_fwd_resp(r);
    pipe_fwd_resp d = pipe_decode_fwd_resp(enc.data(), enc.size(), n_embd, PIPE_HIDDEN_F32);
    CHECK(d.n_tokens == r.n_tokens);
    CHECK(d.hidden   == r.hidden);

    std::vector<uint8_t> trunc(enc.begin(), enc.end() - 1);
    CHECK_THROWS_PROTO(pipe_decode_fwd_resp(trunc.data(), trunc.size(), n_embd, PIPE_HIDDEN_F32),
                       PIPE_ERR_BAD_FRAME);
}

static void test_token_roundtrip() {
    // empty
    {
        pipe_token r; // n_tokens = 0
        std::vector<uint8_t> enc = pipe_encode_token(r);
        pipe_token d = pipe_decode_token(enc.data(), enc.size());
        CHECK(d.token_ids.empty());
    }
    // normal
    {
        pipe_token r;
        r.token_ids = {1, 2999, -5, 128000};
        std::vector<uint8_t> enc = pipe_encode_token(r);
        pipe_token d = pipe_decode_token(enc.data(), enc.size());
        CHECK(d.token_ids == r.token_ids);

        std::vector<uint8_t> trunc(enc.begin(), enc.end() - 1);
        CHECK_THROWS_PROTO(pipe_decode_token(trunc.data(), trunc.size()), PIPE_ERR_BAD_FRAME);
    }
}

static void test_error_roundtrip() {
    pipe_error r;
    r.code = PIPE_ERR_HELLO;
    r.msg  = "stage-set inconsistent: gap at layer 10";
    std::vector<uint8_t> enc = pipe_encode_error(r);
    pipe_error d = pipe_decode_error(enc.data(), enc.size());
    CHECK(d.code == r.code);
    CHECK(d.msg  == r.msg);

    // empty msg
    pipe_error e2; e2.code = PIPE_ERR_GENERIC;
    std::vector<uint8_t> enc2 = pipe_encode_error(e2);
    pipe_error d2 = pipe_decode_error(enc2.data(), enc2.size());
    CHECK(d2.msg.empty());
}

// ---------------------------------------------------------------------------
// endianness: decode a hand-built little-endian HELLO byte string

static void test_endianness_explicit() {
    // role=2 (tail), first=57, last=77, n_layer=78, n_embd=6144, F32, hash=0x01020304
    std::vector<uint8_t> buf;
    put_u32(buf, 2);
    put_i32(buf, 57);
    put_i32(buf, 77);
    put_i32(buf, 78);
    put_i32(buf, 6144);
    put_i32(buf, 0);
    put_u32(buf, 0x01020304u);
    CHECK(buf.size() == pipe_hello::WIRE_SIZE);

    pipe_hello h = pipe_decode_hello(buf.data(), buf.size());
    CHECK(h.role        == PIPE_ROLE_TAIL);
    CHECK(h.layer_first == 57);
    CHECK(h.layer_last  == 77);
    CHECK(h.n_layer     == 78);
    CHECK(h.n_embd      == 6144);
    CHECK(h.hidden_type == PIPE_HIDDEN_F32);
    CHECK(h.model_hash  == 0x01020304u);
}

// ---------------------------------------------------------------------------
// HELLO mismatch matrix: each inconsistent case must be rejected at connect

static pipe_hello good_peer() {
    pipe_hello p;
    p.role        = PIPE_ROLE_TAIL;
    p.layer_first = 57;
    p.layer_last  = 77;
    p.n_layer     = 78;
    p.n_embd      = 6144;
    p.hidden_type = PIPE_HIDDEN_F32;
    p.model_hash  = 0x1234;
    return p;
}

static void test_hello_mismatch_matrix() {
    const int32_t n_layer = 78;
    const int32_t n_embd  = 6144;
    const int32_t ht      = PIPE_HIDDEN_F32;

    // baseline consistent 2-stage set: head [0,56] + tail [57,77] -> accepted
    {
        std::vector<llama_pipeline_stage> stages = {{0, 56}, {57, 77}};
        pipe_hello p = good_peer();
        bool ok = true;
        try {
            pipe_validate_hello(p, n_layer, n_embd, ht, stages);
        } catch (const pipe_protocol_error &) { ok = false; }
        CHECK(ok);
    }

    // gap between bands: head [0,55], tail [57,77] -> layer 56 unowned
    {
        std::vector<llama_pipeline_stage> stages = {{0, 55}, {57, 77}};
        CHECK_THROWS_PROTO(pipe_validate_hello(good_peer(), n_layer, n_embd, ht, stages),
                           PIPE_ERR_HELLO);
    }

    // overlapping bands: head [0,57], tail [57,77] -> layer 57 owned twice
    {
        std::vector<llama_pipeline_stage> stages = {{0, 57}, {57, 77}};
        CHECK_THROWS_PROTO(pipe_validate_hello(good_peer(), n_layer, n_embd, ht, stages),
                           PIPE_ERR_HELLO);
    }

    // peer band not in the agreed set (e.g. a third unexpected stage)
    {
        std::vector<llama_pipeline_stage> stages = {{0, 56}, {57, 77}};
        pipe_hello p = good_peer();
        p.layer_first = 10; p.layer_last = 20; // not in stages
        CHECK_THROWS_PROTO(pipe_validate_hello(p, n_layer, n_embd, ht, stages), PIPE_ERR_HELLO);
    }

    // n_embd mismatch
    {
        std::vector<llama_pipeline_stage> stages = {{0, 56}, {57, 77}};
        CHECK_THROWS_PROTO(pipe_validate_hello(good_peer(), n_layer, /*our_n_embd=*/4096, ht, stages),
                           PIPE_ERR_HELLO);
    }

    // n_layer mismatch
    {
        std::vector<llama_pipeline_stage> stages = {{0, 56}, {57, 77}};
        CHECK_THROWS_PROTO(pipe_validate_hello(good_peer(), /*our_n_layer=*/80, n_embd, ht, stages),
                           PIPE_ERR_HELLO);
    }

    // hidden_type mismatch
    {
        std::vector<llama_pipeline_stage> stages = {{0, 56}, {57, 77}};
        CHECK_THROWS_PROTO(pipe_validate_hello(good_peer(), n_layer, n_embd, PIPE_HIDDEN_F16, stages),
                           PIPE_ERR_HELLO);
    }

    // tail missing: stages do not reach n_layer-1
    {
        std::vector<llama_pipeline_stage> stages = {{0, 56}, {57, 70}};
        pipe_hello p = good_peer();
        p.layer_last = 70;
        CHECK_THROWS_PROTO(pipe_validate_hello(p, n_layer, n_embd, ht, stages), PIPE_ERR_HELLO);
    }

    // unknown hidden_type in the peer HELLO -> rejected before any sizing
    {
        std::vector<llama_pipeline_stage> stages = {{0, 56}, {57, 77}};
        pipe_hello p = good_peer();
        p.hidden_type = 99;
        CHECK_THROWS_PROTO(pipe_validate_hello(p, n_layer, n_embd, 99, stages), PIPE_ERR_HELLO);
    }
}

// ---------------------------------------------------------------------------

int main() {
    test_header_roundtrip();
    test_header_rejects();
    test_hello_roundtrip();
    test_expert_hello_roundtrip();
#ifndef _WIN32
    test_closed_peer_is_transport_error();
#endif
    test_fwd_req_roundtrip();
    test_fwd_req_empty_and_max();
    test_fwd_resp_roundtrip();
    test_token_roundtrip();
    test_error_roundtrip();
    test_endianness_explicit();
    test_hello_mismatch_matrix();

    if (g_failed == 0) {
        std::printf("test-pipe-protocol: all tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "test-pipe-protocol: %d check(s) failed\n", g_failed);
    return 1;
}
