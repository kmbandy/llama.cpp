// Unit tests for the Phase 2 pipeline protocol framing (src/pipeline/pipe-protocol.*).
//
// No model or inference: buffer encode/decode round-trips, malformed-frame
// rejection, the HELLO mismatch matrix, and a closed-peer loopback send.
// Endianness is exercised by decoding hand-built little-endian byte strings.
//
// Standalone build (pipe-protocol.cpp calls ggml_fp32_to_fp16_row /
// ggml_fp16_to_fp32_row for the WP_EXPERT_PARTIAL_DTYPE=f16 path, so this needs
// libggml on the link line too -- point -L/-lggml at your build directory):
//   g++ -std=c++17 -I include -I ggml/include -I src -I src/pipeline
//       tests/test-pipe-protocol.cpp src/pipeline/pipe-protocol.cpp
//       src/pipeline/pipe-transport.cpp src/llama-pipeline.cpp -lggml -o /tmp/t

#include "pipe-protocol.h"
#include "pipe-channel.h"
#include "pipe-transport.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
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
    // PIPE v12 (nextn wire dedup): the IMMEDIATELY PRECEDING version must be rejected
    // too, not just an ancient one. A v11 tail ships the nextn sideband unconditionally
    // and knows no aliasing flag, so a v12 head that negotiated it away would read those
    // bytes as frame body -- and a v11 head cannot declare need at all, which would
    // silently starve draft-mtp of the sideband it verifies against.
    {
        pipe_frame_header h{PIPE_MAGIC, PIPE_VERSION - 1, 1, 0, 0, 0};
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

static void test_channel_fifo_and_harvest() {
    const int port = reserve_transport_port();
    int ready[2];
    CHECK(port > 0);
    CHECK(pipe(ready) == 0);
    const pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        close(ready[0]);
        pipe_socket_ptr server = pipe_socket_t::create_server("127.0.0.1", port);
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
        pipe_frame_type type;
        uint64_t seq_id = 0;
        std::vector<uint8_t> payload;
        if (!pipe_recv_frame(*peer, type, seq_id, payload) || type != PIPE_PING || seq_id != 1) {
            _exit(5);
        }
        if (!pipe_recv_frame(*peer, type, seq_id, payload) || type != PIPE_PONG || seq_id != 0) {
            _exit(6);
        }
        if (!pipe_recv_frame(*peer, type, seq_id, payload) || type != PIPE_TOKEN || seq_id != 2) {
            _exit(7);
        }
        if (!pipe_send_frame(*peer, PIPE_PONG, 1, nullptr, 0) ||
            !pipe_send_frame(*peer, PIPE_TOKEN, 2, nullptr, 0)) {
            _exit(8);
        }
        _exit(0);
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

    {
        pipe_channel::channel channel({ "127.0.0.1", port });
        CHECK(channel.send_request(PIPE_PING, {}) == 1);
        channel.send_frame(PIPE_PONG, 0, {});
        CHECK(channel.send_request(PIPE_TOKEN, {}) == 2);
        channel.flush();

        pipe_channel::received_frame received;
        const std::vector<pipe_channel::channel *> channels = { &channel };
        CHECK(pipe_channel::channel::harvest(channels, received, 5000));
        CHECK(received.source == &channel);
        CHECK(received.type == PIPE_PONG);
        CHECK(received.seq_id == 1);
        CHECK(pipe_channel::channel::harvest(channels, received, 5000));
        CHECK(received.type == PIPE_TOKEN);
        CHECK(received.seq_id == 2);
    }

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

// ---------------------------------------------------------------------------
// EXPERT_PARTIAL dtype tag (2026-08-17, WP_EXPERT_PARTIAL_DTYPE)
//
// Three properties matter here, matching the design requirements this field
// was added for:
//   1. default (PIPE_HIDDEN_F32) is bit-identical to the pre-dtype wire format
//      for the numeric payload -- only the new 4-byte tag itself is added.
//   2. the f16 path round-trips through the REAL ggml conversion (not a
//      hand-rolled approximation) within the documented ~1e-3 relative
//      tolerance, and a spine-side sum built from f16-encoded partials still
//      lands within that tolerance of the all-f32 reference sum.
//   3. an unrecognised dtype tag is rejected as a bad frame, both on encode
//      (a buggy caller) and on decode (a corrupt or hostile peer) -- this is
//      what makes the self-describing tag safe: the spine never blindly
//      trusts a value it cannot interpret.

static void test_expert_partial_f32_default_bit_identical() {
    const int32_t n_embd = 6;
    pipe_expert_partial r;
    CHECK(r.dtype == PIPE_HIDDEN_F32); // struct default, not just a test choice
    r.layer    = 7;
    r.n_tokens = 3;
    r.partial  = {
        0.0f, -0.0f, 1.0f, -1.0f, 3.14159265f, -2.71828f,
        1e-30f, -1e30f, 123456.789f, -0.000123f, 42.0f, -42.0f,
        std::numeric_limits<float>::min(), -std::numeric_limits<float>::min(),
        1.0f, 2.0f, 3.0f, 4.0f,
    };
    CHECK(r.partial.size() == (size_t) r.n_tokens * n_embd);

    std::vector<uint8_t> enc = pipe_encode_expert_partial(r);
    // 4 (layer) + 4 (n_tokens) + 4 (dtype) + 4 bytes/value -- the ONLY new bytes
    // versus the pre-dtype wire format are the 4-byte tag itself.
    CHECK(enc.size() == 12 + r.partial.size() * 4);

    // The bytes AFTER the 12-byte header are a bare memcpy of the f32 array on
    // this (little-endian) host -- i.e. numerically identical to what
    // pipe_encode_expert_partial produced before PIPE_VERSION 13 added the tag.
    CHECK(std::memcmp(enc.data() + 12, r.partial.data(), r.partial.size() * 4) == 0);

    pipe_expert_partial d = pipe_decode_expert_partial(enc.data(), enc.size(), n_embd);
    CHECK(d.dtype    == PIPE_HIDDEN_F32);
    CHECK(d.layer    == r.layer);
    CHECK(d.n_tokens == r.n_tokens);
    // Bit-exact, not approximately-equal: f32-in must equal f32-out exactly.
    CHECK(d.partial.size() == r.partial.size());
    CHECK(std::memcmp(d.partial.data(), r.partial.data(), r.partial.size() * 4) == 0);

    // Truncated payload is still rejected the same way it always was.
    std::vector<uint8_t> trunc(enc.begin(), enc.end() - 1);
    CHECK_THROWS_PROTO(pipe_decode_expert_partial(trunc.data(), trunc.size(), n_embd),
                       PIPE_ERR_BAD_FRAME);
}

static void test_expert_partial_f16_roundtrip_tolerance() {
    const int32_t n_embd = 8;
    pipe_expert_partial r;
    r.dtype    = PIPE_HIDDEN_F16;
    r.layer    = 12;
    r.n_tokens = 2;
    // Representative of a real partial sum: mixed sign, mixed magnitude,
    // including values that do NOT round trip exactly through f16.
    // Deliberately excludes subnormal-range values (below ~6.1e-5): f16's
    // relative precision degrades sharply there by construction (that is what
    // "subnormal" means), which would fail the ~1e-3 relative bound below for
    // reasons that have nothing to do with this code being correct. A partial
    // activation subtotal is never going to be O(1e-5) in practice, so this
    // vector stays in the range the tolerance claim is actually about.
    r.partial = {
        0.0f, 1.0f, -1.0f, 0.1f, -0.1f, 3.14159265f, -3.14159265f, 1000.125f,
        -1000.125f, 0.0009765625f /* exact in f16 */, 12345.6789f, -1.5f,
        65504.0f /* f16 max normal */, -65504.0f, 2.0f, -7.5f,
    };
    CHECK(r.partial.size() == (size_t) r.n_tokens * n_embd);

    std::vector<uint8_t> enc = pipe_encode_expert_partial(r);
    // half the bytes/value versus f32.
    CHECK(enc.size() == 12 + r.partial.size() * 2);

    pipe_expert_partial d = pipe_decode_expert_partial(enc.data(), enc.size(), n_embd);
    CHECK(d.dtype    == PIPE_HIDDEN_F16);
    CHECK(d.layer    == r.layer);
    CHECK(d.n_tokens == r.n_tokens);
    CHECK(d.partial.size() == r.partial.size());

    for (size_t i = 0; i < r.partial.size(); ++i) {
        // The decode MUST use the real ggml fp16<->fp32 conversion, not a
        // hand-rolled approximation: the per-element reference computed
        // directly from ggml_fp32_to_fp16/ggml_fp16_to_fp32 must match the
        // bulk-row path bit-for-bit.
        const float reference = ggml_fp16_to_fp32(ggml_fp32_to_fp16(r.partial[i]));
        if (reference != d.partial[i]) {
            std::fprintf(stderr,
                "FAIL %s:%d: element %zu: bulk row path (%.9g) != scalar ggml "
                "reference (%.9g) for input %.9g\n",
                __FILE__, __LINE__, i, (double) d.partial[i], (double) reference,
                (double) r.partial[i]);
            ++g_failed;
        }
        // Documented tolerance: ~1e-3 relative (f16 has an 11-bit mantissa, so
        // worst case is ~2^-11 =~ 4.9e-4 relative -- 1e-3 gives headroom).
        const double a   = (double) r.partial[i];
        const double b   = (double) d.partial[i];
        const double den = std::max(std::fabs(a), 1e-12);
        const double rel = std::fabs(a - b) / den;
        if (rel > 1e-3) {
            std::fprintf(stderr,
                "FAIL %s:%d: element %zu relative error %.6g exceeds 1e-3 "
                "(orig=%.9g decoded=%.9g)\n",
                __FILE__, __LINE__, i, rel, a, b);
            ++g_failed;
        }
    }

    // SPINE-SIDE SUM CHECK: two workers each cover half a layer's experts.
    // Worker A sends its subtotal as f16 (opted in via WP_EXPERT_PARTIAL_DTYPE);
    // worker B stays on the f32 default. The spine's scatter_add just adds
    // whatever pipe_decode_expert_partial() handed back -- see the note in
    // pipe-expert-dispatcher.cpp's receive_partial() -- so this reproduces that
    // exact shape without spinning up sockets or workers.
    pipe_expert_partial worker_b;
    worker_b.dtype    = PIPE_HIDDEN_F32;
    worker_b.layer    = r.layer;
    worker_b.n_tokens = r.n_tokens;
    worker_b.partial.assign(r.partial.size(), 0.0f);
    for (size_t i = 0; i < worker_b.partial.size(); ++i) {
        worker_b.partial[i] = (float) i * 0.5f - 3.0f;
    }
    std::vector<uint8_t> enc_b = pipe_encode_expert_partial(worker_b);
    pipe_expert_partial  dec_b = pipe_decode_expert_partial(enc_b.data(), enc_b.size(), n_embd);
    CHECK(dec_b.dtype == PIPE_HIDDEN_F32);

    for (size_t i = 0; i < r.partial.size(); ++i) {
        const double wire_sum = (double) d.partial[i] + (double) dec_b.partial[i];
        const double true_sum = (double) r.partial[i] + (double) worker_b.partial[i];
        const double den      = std::max(std::fabs(true_sum), 1e-12);
        const double rel      = std::fabs(wire_sum - true_sum) / den;
        if (rel > 1e-3) {
            std::fprintf(stderr,
                "FAIL %s:%d: scatter_add-equivalent sum element %zu relative "
                "error %.6g exceeds 1e-3 (wire=%.9g true=%.9g)\n",
                __FILE__, __LINE__, i, rel, wire_sum, true_sum);
            ++g_failed;
        }
    }
}

static void test_expert_partial_dtype_rejects() {
    const int32_t n_embd = 4;

    // encode: unknown dtype is rejected before any bytes are written.
    {
        pipe_expert_partial r;
        r.layer    = 0;
        r.n_tokens = 1;
        r.dtype    = 99; // not PIPE_HIDDEN_F32 or PIPE_HIDDEN_F16
        r.partial.assign((size_t) n_embd, 1.0f);
        CHECK_THROWS_PROTO(pipe_encode_expert_partial(r), PIPE_ERR_BAD_FRAME);
    }

    // decode: a hand-built frame with a garbage dtype word is rejected, not
    // silently misread as f32 or f16. This is the "mismatch cannot corrupt a
    // sum" property in its most direct form -- an invalid tag is refused
    // rather than guessed at.
    {
        std::vector<uint8_t> buf;
        put_i32(buf, 0); // layer
        put_u32(buf, 1); // n_tokens
        put_i32(buf, 99); // dtype: garbage
        for (int i = 0; i < n_embd; ++i) put_u32(buf, 0);
        CHECK_THROWS_PROTO(pipe_decode_expert_partial(buf.data(), buf.size(), n_embd),
                           PIPE_ERR_BAD_FRAME);
    }
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
// expert prefetch hint (PIPE_VERSION 5)

static void test_expert_prefetch_hint_roundtrip() {
    pipe_expert_prefetch_hint h;
    h.layer      = 2;
    h.expert_ids = { 0, 7, 8, 84, 255 };

    const std::vector<uint8_t> enc = pipe_encode_expert_prefetch_hint(h);
    CHECK(enc.size() == 12 + 5 * 4);

    const pipe_expert_prefetch_hint d =
        pipe_decode_expert_prefetch_hint(enc.data(), enc.size());
    CHECK(d.layer == h.layer);
    CHECK(d.provenance == h.provenance);
    CHECK(d.expert_ids == h.expert_ids);

    // Layer 0 is a real hash layer, not a sentinel -- it must encode.
    pipe_expert_prefetch_hint zero;
    zero.layer      = 0;
    zero.expert_ids = { 3 };
    const std::vector<uint8_t> zenc = pipe_encode_expert_prefetch_hint(zero);
    CHECK(pipe_decode_expert_prefetch_hint(zenc.data(), zenc.size()).layer == 0);
}

static void test_expert_prefetch_hint_rejects() {
    // Encode side: the caller must not manufacture a frame the worker will
    // reject, because a hint has no response and the failure would be silent.
    {
        pipe_expert_prefetch_hint h;
        h.layer      = -1;
        h.expert_ids = { 1 };
        CHECK_THROWS_PROTO(pipe_encode_expert_prefetch_hint(h), PIPE_ERR_BAD_FRAME);
    }
    {
        pipe_expert_prefetch_hint h;   // empty set is "do not send", not a valid frame
        h.layer = 0;
        CHECK_THROWS_PROTO(pipe_encode_expert_prefetch_hint(h), PIPE_ERR_BAD_FRAME);
    }
    {
        pipe_expert_prefetch_hint h;
        h.layer      = 0;
        h.expert_ids = { 5, 5 };       // duplicate: would double-count a page-in
        CHECK_THROWS_PROTO(pipe_encode_expert_prefetch_hint(h), PIPE_ERR_BAD_FRAME);
    }
    {
        pipe_expert_prefetch_hint h;
        h.layer      = 0;
        h.expert_ids = { 9, 2 };       // descending: loses the sequential read order
        CHECK_THROWS_PROTO(pipe_encode_expert_prefetch_hint(h), PIPE_ERR_BAD_FRAME);
    }
    {
        pipe_expert_prefetch_hint h;
        h.layer      = 0;
        h.expert_ids = { -1 };
        CHECK_THROWS_PROTO(pipe_encode_expert_prefetch_hint(h), PIPE_ERR_BAD_FRAME);
    }

    // Decode side: hand-built little-endian payloads, since a hostile or stale
    // peer does not go through our encoder.
    {
        std::vector<uint8_t> v;
        put_i32(v, 0);
        put_u32(v, 0);                 // n_experts = 0
        CHECK_THROWS_PROTO(pipe_decode_expert_prefetch_hint(v.data(), v.size()),
                           PIPE_ERR_BAD_FRAME);
    }
    {
        std::vector<uint8_t> v;
        put_i32(v, 0);
        put_u32(v, 3);                 // claims 3, carries 2
        put_i32(v, 1);
        put_i32(v, 2);
        CHECK_THROWS_PROTO(pipe_decode_expert_prefetch_hint(v.data(), v.size()),
                           PIPE_ERR_BAD_FRAME);
    }
    {
        std::vector<uint8_t> v;
        put_i32(v, 1);
        put_u32(v, 2);
        put_i32(v, 4);
        put_i32(v, 4);                 // duplicate on the wire
        CHECK_THROWS_PROTO(pipe_decode_expert_prefetch_hint(v.data(), v.size()),
                           PIPE_ERR_BAD_FRAME);
    }
    {
        std::vector<uint8_t> v;
        put_i32(v, 1);
        put_u32(v, 2);
        put_i32(v, 4);
        put_i32(v, 1);                 // descending on the wire
        CHECK_THROWS_PROTO(pipe_decode_expert_prefetch_hint(v.data(), v.size()),
                           PIPE_ERR_BAD_FRAME);
    }
    {
        std::vector<uint8_t> v;
        put_i32(v, 0);
        put_u32(v, 1);
        put_i32(v, -3);                // negative id
        CHECK_THROWS_PROTO(pipe_decode_expert_prefetch_hint(v.data(), v.size()),
                           PIPE_ERR_BAD_FRAME);
    }
    {
        std::vector<uint8_t> v;
        put_i32(v, 0);                 // header only, no count
        CHECK_THROWS_PROTO(pipe_decode_expert_prefetch_hint(v.data(), v.size()),
                           PIPE_ERR_BAD_FRAME);
    }
}

// ---------------------------------------------------------------------------

static void test_segment_roundtrip() {
    pipe_segment_hello hello;
    hello.segment_id = 2;
    hello.layer_first = 24;
    hello.layer_last = 35;
    hello.model_identity_sha256 = "sha256:dense-segment-model";
    hello.n_embd = 8;
    hello.n_vocab = 32;
    hello.capabilities = PIPE_SEGMENT_CAP_FWD | PIPE_SEGMENT_CAP_KV_TRIM |
                         PIPE_SEGMENT_CAP_RECURRENT;
    hello.cache_epoch = 17;
    hello.terminal_kind = PIPE_SEGMENT_TERMINAL_HIDDEN;

    const std::vector<uint8_t> hello_enc = pipe_encode_segment_hello(hello);
    const pipe_segment_hello hello_dec =
        pipe_decode_segment_hello(hello_enc.data(), hello_enc.size());
    CHECK(hello_dec.segment_id == hello.segment_id);
    CHECK(hello_dec.layer_first == hello.layer_first);
    CHECK(hello_dec.layer_last == hello.layer_last);
    CHECK(hello_dec.model_identity_sha256 == hello.model_identity_sha256);
    CHECK(hello_dec.n_embd == hello.n_embd);
    CHECK(hello_dec.n_vocab == hello.n_vocab);
    CHECK(hello_dec.capabilities == hello.capabilities);
    CHECK(hello_dec.cache_epoch == hello.cache_epoch);
    CHECK(hello_dec.terminal_kind == hello.terminal_kind);

    // logits-on-head: an unknown terminal kind must be rejected, not defaulted
    {
        pipe_segment_hello bad = hello;
        bad.terminal_kind = 99;
        CHECK_THROWS_PROTO(pipe_encode_segment_hello(bad), PIPE_ERR_HELLO);
    }

    // interior taps: a request carrying taps must round-trip exactly
    {
        pipe_segment_hello taps = hello;
        taps.tap_layers = { 26, 31 };
        const std::vector<uint8_t> enc = pipe_encode_segment_hello(taps);
        const pipe_segment_hello dec = pipe_decode_segment_hello(enc.data(), enc.size());
        CHECK(dec.tap_layers == taps.tap_layers);
        // the rest of the payload must survive the new variable-length tail
        CHECK(dec.model_identity_sha256 == taps.model_identity_sha256);
        CHECK(dec.terminal_kind == taps.terminal_kind);
        CHECK(dec.cache_epoch == taps.cache_epoch);
    }

    // interior taps: outside the declared band is unserveable, so reject it
    {
        pipe_segment_hello bad = hello;
        bad.tap_layers = { 99 };
        CHECK_THROWS_PROTO(pipe_encode_segment_hello(bad), PIPE_ERR_HELLO);
    }

    // interior taps: the ascending/unique order is the wire contract -- the response
    // carries no per-block layer id, so both peers derive the mapping from this list
    {
        pipe_segment_hello bad = hello;
        bad.tap_layers = { 31, 26 };
        CHECK_THROWS_PROTO(pipe_encode_segment_hello(bad), PIPE_ERR_HELLO);

        bad.tap_layers = { 26, 26 };
        CHECK_THROWS_PROTO(pipe_encode_segment_hello(bad), PIPE_ERR_HELLO);
    }

    // nextn sideband: the need flag must survive the tap list, which sits between it and
    // the fixed header, and it must round-trip in BOTH states -- "not needed" is the
    // production default and is the whole point of the field
    {
        pipe_segment_hello need = hello;
        need.nextn_need = 1;
        need.tap_layers = { 26, 31 };
        const std::vector<uint8_t> enc = pipe_encode_segment_hello(need);
        const pipe_segment_hello dec = pipe_decode_segment_hello(enc.data(), enc.size());
        CHECK(dec.nextn_need == 1);
        CHECK(dec.tap_layers == need.tap_layers);
        CHECK(dec.model_identity_sha256 == need.model_identity_sha256);

        CHECK(pipe_decode_segment_hello(hello_enc.data(), hello_enc.size()).nextn_need == 0);

        // a boolean on the wire, not an enum: anything but 0/1 is a peer we do not
        // understand, and guessing "truthy" would ship a duplicate run per token
        pipe_segment_hello bad = hello;
        bad.nextn_need = 2;
        CHECK_THROWS_PROTO(pipe_encode_segment_hello(bad), PIPE_ERR_HELLO);
    }

    const pipe_segment_hello_ack hello_ack{ PIPE_SEGMENT_HELLO_VERSION, true, 32, 16,
                                            PIPE_SEGMENT_TERMINAL_HIDDEN, 8, "" };
    const std::vector<uint8_t> hello_ack_enc = pipe_encode_segment_hello_ack(hello_ack);
    const pipe_segment_hello_ack hello_ack_dec =
        pipe_decode_segment_hello_ack(hello_ack_enc.data(), hello_ack_enc.size());
    CHECK(hello_ack_dec.accepted);
    CHECK(hello_ack_dec.terminal_kind == PIPE_SEGMENT_TERMINAL_HIDDEN);
    CHECK(hello_ack_dec.terminal_width == 8);

    const pipe_segment_hello_ack logits_ack{ PIPE_SEGMENT_HELLO_VERSION, true, 32, 16,
                                             PIPE_SEGMENT_TERMINAL_LOGITS, 32, "" };
    const std::vector<uint8_t> logits_ack_enc = pipe_encode_segment_hello_ack(logits_ack);
    const pipe_segment_hello_ack logits_ack_dec =
        pipe_decode_segment_hello_ack(logits_ack_enc.data(), logits_ack_enc.size());
    CHECK(logits_ack_dec.terminal_kind == PIPE_SEGMENT_TERMINAL_LOGITS);
    CHECK(logits_ack_dec.terminal_width == 32);

    // interior taps: the ACK echo and its width must round-trip alongside `reason`,
    // which is the other variable-length tail in this payload
    {
        pipe_segment_hello_ack tap_ack = hello_ack;
        tap_ack.tap_layers = { 26, 31 };
        tap_ack.tap_width  = 8;
        const std::vector<uint8_t> enc = pipe_encode_segment_hello_ack(tap_ack);
        const pipe_segment_hello_ack dec =
            pipe_decode_segment_hello_ack(enc.data(), enc.size());
        CHECK(dec.tap_layers == tap_ack.tap_layers);
        CHECK(dec.tap_width == 8);
        CHECK(dec.terminal_width == 8);
        CHECK(dec.reason.empty());

        // a tap list with no width cannot be sliced, so it must not encode
        pipe_segment_hello_ack bad = tap_ack;
        bad.tap_width = 0;
        CHECK_THROWS_PROTO(pipe_encode_segment_hello_ack(bad), PIPE_ERR_HELLO);
    }

    // nextn sideband: the ACK's answer must round-trip past BOTH variable-length tails
    // (taps, then reason). The default ACK above answers 0, which is the arm where the
    // sideband costs nothing.
    {
        CHECK(hello_ack_dec.nextn_width == 0);

        pipe_segment_hello_ack nextn_ack = hello_ack;
        nextn_ack.nextn_width = 8;
        nextn_ack.tap_layers  = { 26, 31 };
        nextn_ack.tap_width   = 8;
        const std::vector<uint8_t> enc = pipe_encode_segment_hello_ack(nextn_ack);
        const pipe_segment_hello_ack dec =
            pipe_decode_segment_hello_ack(enc.data(), enc.size());
        CHECK(dec.nextn_width == 8);
        CHECK(dec.tap_width == 8);
        CHECK(dec.tap_layers == nextn_ack.tap_layers);
        CHECK(dec.terminal_width == 8);
        CHECK(dec.reason.empty());
    }

    pipe_segment_fwd_req request;
    request.session_id = 9;
    request.seq_id = 42;
    request.n_tokens = 2;
    request.n_pos_per_token = 2;
    request.n_seqs = 1;
    request.positions = { 4, 0, 5, 0 };
    request.seq_token_counts = { 2 };
    request.activations = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    };
    const std::vector<uint8_t> request_enc = pipe_encode_segment_fwd_req(request, 8);
    const pipe_segment_fwd_req request_dec =
        pipe_decode_segment_fwd_req(request_enc.data(), request_enc.size(), 8);
    CHECK(request_dec.session_id == request.session_id);
    CHECK(request_dec.seq_id == request.seq_id);
    CHECK(request_dec.positions == request.positions);
    CHECK(request_dec.seq_token_counts == request.seq_token_counts);
    CHECK(request_dec.activations == request.activations);

    pipe_segment_fwd_resp response;
    response.session_id = request.session_id;
    response.seq_id = request.seq_id;
    response.n_tokens = request.n_tokens;
    response.output_width = 8;
    response.nextn_width = 0;
    response.activations = request.activations;
    const std::vector<uint8_t> response_enc = pipe_encode_segment_fwd_resp(response);
    const pipe_segment_fwd_resp response_dec =
        pipe_decode_segment_fwd_resp(response_enc.data(), response_enc.size(), 8, 0, 0, 0);
    CHECK(response_dec.session_id == response.session_id);
    CHECK(response_dec.seq_id == response.seq_id);
    CHECK(response_dec.activations == response.activations);

    // interior taps ride the forward response as N concatenated [n_tokens, tap_width]
    // blocks, after `nextn`. n_tokens = 2 and tap_width = 8 here, two taps.
    {
        pipe_segment_fwd_resp taps = response;
        taps.tap_width = 8;
        taps.n_taps    = 2;
        taps.taps.assign((size_t) taps.n_tokens * taps.tap_width * taps.n_taps, 0.0f);
        for (size_t i = 0; i < taps.taps.size(); ++i) {
            taps.taps[i] = (float) (100 + i);
        }
        const std::vector<uint8_t> enc = pipe_encode_segment_fwd_resp(taps);
        const pipe_segment_fwd_resp dec =
            pipe_decode_segment_fwd_resp(enc.data(), enc.size(), 8, 0, 8, 2);
        CHECK(dec.tap_width == 8);
        CHECK(dec.n_taps == 2);
        CHECK(dec.taps == taps.taps);
        CHECK(dec.activations == taps.activations);

        // A peer that quietly stopped extracting is the failure this feature must never
        // allow: it changes no verified token, so only the decoder can catch it.
        CHECK_THROWS_PROTO(
            pipe_decode_segment_fwd_resp(enc.data(), enc.size(), 8, 0, 8, 1),
            PIPE_ERR_BAD_FRAME);
        CHECK_THROWS_PROTO(
            pipe_decode_segment_fwd_resp(enc.data(), enc.size(), 8, 0, 0, 0),
            PIPE_ERR_BAD_FRAME);
    }

    // NEXTN WIRE DEDUP. This block is the whole point of PIPE v12, so it asserts BYTES,
    // not just field values: the saving is invisible in a field-by-field round-trip.
    {
        // (a) the un-deduped shape, for the byte baseline: nextn is a second full run
        pipe_segment_fwd_resp both = response;
        both.nextn_width = 8;
        both.nextn.assign((size_t) both.n_tokens * 8, 7.0f);
        const std::vector<uint8_t> both_enc = pipe_encode_segment_fwd_resp(both);
        const pipe_segment_fwd_resp both_dec =
            pipe_decode_segment_fwd_resp(both_enc.data(), both_enc.size(), 8, 8, 0, 0);
        CHECK(both_dec.nextn_aliased == 0);
        CHECK(both_dec.nextn == both.nextn);

        // (b) aliased: same negotiated width, but the run is OFF THE WIRE. The frame
        // must shrink by exactly one [n_tokens, nextn_width] f32 block.
        pipe_segment_fwd_resp aliased = response;
        aliased.nextn_width   = 8;
        aliased.nextn_aliased = 1;
        aliased.nextn.clear();
        const std::vector<uint8_t> aliased_enc = pipe_encode_segment_fwd_resp(aliased);
        CHECK(both_enc.size() - aliased_enc.size() ==
              (size_t) response.n_tokens * 8 * sizeof(float));

        // the decoder reports the flag and leaves `nextn` empty -- reconstruction is the
        // client's job, so encode/decode stays a pure byte transform
        const pipe_segment_fwd_resp aliased_dec =
            pipe_decode_segment_fwd_resp(aliased_enc.data(), aliased_enc.size(), 8, 8, 0, 0);
        CHECK(aliased_dec.nextn_aliased == 1);
        CHECK(aliased_dec.nextn.empty());
        CHECK(aliased_dec.nextn_width == 8);
        CHECK(aliased_dec.activations == response.activations);

        // (c) no need negotiated => zero nextn bytes, which is the dspark/no-spec arm.
        // Compare against the baseline to prove the run really is absent.
        CHECK(both_enc.size() - response_enc.size() ==
              (size_t) response.n_tokens * 8 * sizeof(float));

        // (d) a tail that ignored a nextn_width=0 negotiation and appended the run
        // anyway must fail the decode, not silently feed the head 8 floats of taps
        CHECK_THROWS_PROTO(
            pipe_decode_segment_fwd_resp(both_enc.data(), both_enc.size(), 8, 0, 0, 0),
            PIPE_ERR_BAD_FRAME);

        // (e) aliasing is only representable when the two runs have the same shape and a
        // sideband was actually negotiated -- otherwise the head would rebuild a buffer
        // of the wrong width, or one nobody asked for
        pipe_segment_fwd_resp bad = aliased;
        bad.nextn_width = 4;   // != output_width 8
        CHECK_THROWS_PROTO(pipe_encode_segment_fwd_resp(bad), PIPE_ERR_BAD_FRAME);

        bad = aliased;
        bad.nextn_width = 0;   // aliased with no negotiated sideband
        CHECK_THROWS_PROTO(pipe_encode_segment_fwd_resp(bad), PIPE_ERR_BAD_FRAME);

        // and the flag must not be set while the run is still attached: that would
        // encode a length the decoder cannot reproduce
        bad = aliased;
        bad.nextn.assign((size_t) bad.n_tokens * 8, 7.0f);
        CHECK_THROWS_PROTO(pipe_encode_segment_fwd_resp(bad), PIPE_ERR_BAD_FRAME);
    }

    const pipe_segment_ctrl trim = {
        PIPE_SEGMENT_CTRL_VERSION, PIPE_SEGMENT_CTRL_KV_TRIM, 9, 17, 11, "",
    };
    const std::vector<uint8_t> trim_enc = pipe_encode_segment_ctrl(trim);
    const pipe_segment_ctrl trim_dec = pipe_decode_segment_ctrl(trim_enc.data(), trim_enc.size());
    CHECK(trim_dec.control == PIPE_SEGMENT_CTRL_KV_TRIM);
    CHECK(trim_dec.session_id == trim.session_id);
    CHECK(trim_dec.cache_epoch == trim.cache_epoch);
    CHECK(trim_dec.n_past == trim.n_past);

    const pipe_segment_ctrl reuse = {
        PIPE_SEGMENT_CTRL_VERSION, PIPE_SEGMENT_CTRL_PROMPT_REUSE, 9, 17, 11,
        "sha256:prompt-prefix",
    };
    const std::vector<uint8_t> reuse_enc = pipe_encode_segment_ctrl(reuse);
    CHECK(pipe_decode_segment_ctrl(reuse_enc.data(), reuse_enc.size()).prompt_identity_sha256 ==
          reuse.prompt_identity_sha256);

    const pipe_segment_ctrl_ack ack = {
        PIPE_SEGMENT_CTRL_ACK_VERSION, PIPE_SEGMENT_CTRL_KV_TRIM, 9, 17,
        PIPE_SEGMENT_CTRL_APPLIED, 11,
    };
    const std::vector<uint8_t> ack_enc = pipe_encode_segment_ctrl_ack(ack);
    const pipe_segment_ctrl_ack ack_dec = pipe_decode_segment_ctrl_ack(ack_enc.data(), ack_enc.size());
    CHECK(ack_dec.control == ack.control);
    CHECK(ack_dec.session_id == ack.session_id);
    CHECK(ack_dec.cache_epoch == ack.cache_epoch);
    CHECK(ack_dec.status == ack.status);

    // Old-version rejection at the PAYLOAD level, not just the frame header. The segment
    // payloads carry their own versions precisely so a mixed-version pair fails on the
    // first frame that matters rather than mis-slicing it.
    {
        pipe_segment_hello old_hello = hello;
        old_hello.version = PIPE_SEGMENT_HELLO_VERSION - 1;
        CHECK_THROWS_PROTO(pipe_encode_segment_hello(old_hello), PIPE_ERR_HELLO);

        pipe_segment_hello_ack old_ack = hello_ack;
        old_ack.version = PIPE_SEGMENT_HELLO_VERSION - 1;
        CHECK_THROWS_PROTO(pipe_encode_segment_hello_ack(old_ack), PIPE_ERR_HELLO);

        pipe_segment_fwd_resp old_resp = response;
        old_resp.version = PIPE_SEGMENT_FWD_VERSION - 1;
        CHECK_THROWS_PROTO(pipe_encode_segment_fwd_resp(old_resp), PIPE_ERR_BAD_FRAME);
    }

    pipe_segment_fwd_req bad_request = request;
    bad_request.seq_token_counts = { 1 };
    CHECK_THROWS_PROTO(pipe_encode_segment_fwd_req(bad_request, 8), PIPE_ERR_BAD_FRAME);
    const pipe_segment_ctrl bad_reset = {
        PIPE_SEGMENT_CTRL_VERSION, PIPE_SEGMENT_CTRL_RESET, 9, 17, 1, "",
    };
    CHECK_THROWS_PROTO(pipe_encode_segment_ctrl(bad_reset), PIPE_ERR_BAD_FRAME);
}

// ---------------------------------------------------------------------------

int main() {
    test_header_roundtrip();
    test_header_rejects();
    test_hello_roundtrip();
    test_expert_hello_roundtrip();
#ifndef _WIN32
    test_closed_peer_is_transport_error();
    test_channel_fifo_and_harvest();
#endif
    test_fwd_req_roundtrip();
    test_fwd_req_empty_and_max();
    test_fwd_resp_roundtrip();
    test_expert_partial_f32_default_bit_identical();
    test_expert_partial_f16_roundtrip_tolerance();
    test_expert_partial_dtype_rejects();
    test_token_roundtrip();
    test_error_roundtrip();
    test_endianness_explicit();
    test_hello_mismatch_matrix();
    test_expert_prefetch_hint_roundtrip();
    test_expert_prefetch_hint_rejects();
    test_segment_roundtrip();

    if (g_failed == 0) {
        std::printf("test-pipe-protocol: all tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "test-pipe-protocol: %d check(s) failed\n", g_failed);
    return 1;
}
