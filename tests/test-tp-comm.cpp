// Cross-host tensor-parallel exchange: loopback correctness and the fixed add order.
//
// Two threads, one real TCP connection over 127.0.0.1, N partial exchanges. Both sides must end
// every exchange holding bit-identical buffers, and that value must be the sum of the two partials
// computed in the order S_rank0 + S_rank1 regardless of which side is executing.
//
// No model, no GPU. The port is chosen by the OS-free range below and retried on collision.

#include "pipe-tp-comm.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n  ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                    \
            fprintf(stderr, "\n");                                           \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------------------------
// 1. THE DETERMINISM RULE, with no socket in the way.
//
//    S_total = S_rank0 + S_rank1, in that operand order, on BOTH ranks. The two role assignments
//    must produce BIT-identical results - not almost-equal, identical - because the whole
//    cross-host arm is gated on producing the same tokens as a single process at temperature 0.
// ---------------------------------------------------------------------------------------------
static void test_fixed_order_add_is_bit_identical() {
    std::mt19937 rng(1234);
    // deliberately nasty magnitudes: float addition is commutative but NOT associative, so if the
    // implementation ever grows a third operand or a fused form this is what catches it
    std::uniform_real_distribution<float> big(-1e18f, 1e18f);
    std::uniform_real_distribution<float> small(-1e-18f, 1e-18f);
    std::uniform_real_distribution<float> normal(-4.0f, 4.0f);

    const size_t n = 5120 * 7;
    std::vector<float> s0(n), s1(n);
    for (size_t i = 0; i < n; i++) {
        switch (i % 3) {
            case 0:  s0[i] = big(rng);    s1[i] = small(rng);  break;
            case 1:  s0[i] = small(rng);  s1[i] = big(rng);    break;
            default: s0[i] = normal(rng); s1[i] = normal(rng); break;
        }
    }
    // a few exact edge values
    s0[0] = 0.0f;              s1[0] = -0.0f;
    s0[1] = INFINITY;          s1[1] = 1.0f;
    s0[2] = 1.0f;              s1[2] = -1.0f;
    s0[3] = 1.0f;              s1[3] = std::ldexp(1.0f, -30); // rounds away

    // rank 0's view: local = s0, peer = s1
    std::vector<float> as_rank0 = s0;
    pipe_tp_add_fixed_order(as_rank0.data(), s1.data(), n, /*local_is_rank0 =*/ true);

    // rank 1's view: local = s1, peer = s0
    std::vector<float> as_rank1 = s1;
    pipe_tp_add_fixed_order(as_rank1.data(), s0.data(), n, /*local_is_rank0 =*/ false);

    size_t n_diff = 0;
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&as_rank0[i], &as_rank1[i], sizeof(float)) != 0) {
            n_diff++;
            if (n_diff <= 4) {
                fprintf(stderr, "  index %zu: rank0 view %a, rank1 view %a (s0 %a, s1 %a)\n",
                        i, (double) as_rank0[i], (double) as_rank1[i], (double) s0[i], (double) s1[i]);
            }
        }
    }
    CHECK(n_diff == 0, "%zu of %zu values differ between the two rank views of the same sum", n_diff, n);

    // and the value is right
    for (size_t i = 4; i < n; i++) {
        const float expect = s0[i] + s1[i];
        CHECK(memcmp(&as_rank0[i], &expect, sizeof(float)) == 0 || i < 4,
              "index %zu: got %a, expected %a", i, (double) as_rank0[i], (double) expect);
        if (g_failures > 8) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 2. LOOPBACK EXCHANGE.
// ---------------------------------------------------------------------------------------------

struct arm_result {
    bool                connected = false;
    bool                ok        = true;
    std::vector<float>  last;
    pipe_tp_stats       stats;
};

// deterministic per-(rank, round, index) partial, so both threads can predict the expected total
static float partial_value(int rank, int round, size_t i) {
    const uint32_t h = (uint32_t) (rank * 2654435761u + round * 40503u + (uint32_t) i * 2246822519u);
    return ((float) (h % 20011u) - 10005.0f) / 977.0f;
}

static void run_arm(bool is_rank0, const std::string & host, int port,
                    const std::vector<size_t> & widths, size_t max_values, arm_result * out) {
    std::unique_ptr<pipe_tp_comm> comm = is_rank0
        ? pipe_tp_comm::listen (host, port, max_values, 10000)
        : pipe_tp_comm::connect(host, port, max_values, 10000);
    if (!comm) {
        out->ok = false;
        return;
    }
    out->connected = true;

    const int rank = is_rank0 ? 0 : 1;
    std::vector<float> buf(max_values);
    for (size_t r = 0; r < widths.size(); r++) {
        const size_t n = widths[r];
        for (size_t i = 0; i < n; i++) {
            buf[i] = partial_value(rank, (int) r, i);
        }
        if (!comm->exchange_add(buf.data(), n, is_rank0)) {
            fprintf(stderr, "  rank %d: exchange %zu failed\n", rank, r);
            out->ok = false;
            return;
        }
        for (size_t i = 0; i < n; i++) {
            const float expect = partial_value(0, (int) r, i) + partial_value(1, (int) r, i);
            if (memcmp(&buf[i], &expect, sizeof(float)) != 0) {
                fprintf(stderr, "  rank %d round %zu index %zu: got %a expected %a\n",
                        rank, r, i, (double) buf[i], (double) expect);
                out->ok = false;
                return;
            }
        }
        out->last.assign(buf.begin(), buf.begin() + n);
    }
    out->stats = comm->stats();
}

static void test_loopback_exchange() {
    // Widths that matter: a single token, a 7-wide MTP verify batch, and one wide enough that the
    // payload exceeds a default socket buffer - the case a blocking send-then-recv would deadlock
    // on and the poll loop must not.
    const size_t n_embd = 5120;
    const std::vector<size_t> widths = {
        n_embd * 1,
        n_embd * 3,
        n_embd * 7,
        n_embd * 1,
        n_embd * 512, // 10 MiB payload each way, simultaneously, in both directions
    };
    size_t max_values = 0;
    for (size_t w : widths) {
        max_values = std::max(max_values, w);
    }

    const std::string host = "127.0.0.1";
    bool any_connected = false;
    for (int port = 47811; port < 47831 && !any_connected; port++) {
        arm_result r0, r1;
        std::thread t0([&] { run_arm(true,  host, port, widths, max_values, &r0); });
        // give the listener a moment to bind before the connector's first attempt; connect()
        // retries a refusal for up to 10 s anyway, this just keeps the log quiet
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::thread t1([&] { run_arm(false, host, port, widths, max_values, &r1); });
        t0.join();
        t1.join();

        if (!r0.connected || !r1.connected) {
            continue; // port busy, try the next one
        }
        any_connected = true;

        CHECK(r0.ok, "rank 0 arm failed");
        CHECK(r1.ok, "rank 1 arm failed");
        CHECK(r0.last.size() == r1.last.size(), "final buffers differ in size");
        if (r0.last.size() == r1.last.size() && !r0.last.empty()) {
            CHECK(memcmp(r0.last.data(), r1.last.data(), r0.last.size() * sizeof(float)) == 0,
                  "the two ranks ended with different bytes");
        }
        CHECK(r0.stats.n_exchanges == widths.size(), "rank 0 counted %llu exchanges, expected %zu",
              (unsigned long long) r0.stats.n_exchanges, widths.size());
        CHECK(r1.stats.n_exchanges == widths.size(), "rank 1 counted %llu exchanges, expected %zu",
              (unsigned long long) r1.stats.n_exchanges, widths.size());

        uint64_t payload = 0;
        for (size_t w : widths) {
            payload += w * sizeof(float) + sizeof(pipe_tp_frame_hdr);
        }
        CHECK(r0.stats.bytes_sent  == payload, "rank 0 sent %llu bytes, expected %llu",
              (unsigned long long) r0.stats.bytes_sent, (unsigned long long) payload);
        CHECK(r0.stats.bytes_recvd == payload, "rank 0 received %llu bytes, expected %llu",
              (unsigned long long) r0.stats.bytes_recvd, (unsigned long long) payload);

        printf("  loopback on port %d: %llu exchanges, %llu B each way, %.3f ms blocked total\n",
               port,
               (unsigned long long) r0.stats.n_exchanges,
               (unsigned long long) r0.stats.bytes_sent,
               r0.stats.ns_blocked / 1e6);
    }
    CHECK(any_connected, "could not establish a loopback connection on any port in 47811..47830");
}

// ---------------------------------------------------------------------------------------------
// 3. Frame header shape: M3's control channel shares this connection, so the type byte and the
//    length must be where a reader expects them and the header must not grow.
// ---------------------------------------------------------------------------------------------
static void test_frame_header_layout() {
    CHECK(sizeof(pipe_tp_frame_hdr) == 16, "header is %zu bytes", sizeof(pipe_tp_frame_hdr));
    CHECK(offsetof(pipe_tp_frame_hdr, magic)         == 0,  "magic moved");
    CHECK(offsetof(pipe_tp_frame_hdr, type)          == 4,  "type moved");
    CHECK(offsetof(pipe_tp_frame_hdr, dtype)         == 5,  "dtype moved");
    CHECK(offsetof(pipe_tp_frame_hdr, payload_bytes) == 12, "payload_bytes moved");
    CHECK(PIPE_TP_MSG_REDUCE != PIPE_TP_MSG_HELLO && PIPE_TP_MSG_HELLO != PIPE_TP_MSG_DECODE,
          "reserved M3 message types collide with REDUCE");
}

static void test_parse_peer() {
    std::string host;
    int port = 0;
    CHECK(pipe_tp_comm::parse_peer("192.168.1.33:8899", &host, &port) && host == "192.168.1.33" && port == 8899,
          "failed to parse a plain host:port");
    CHECK(pipe_tp_comm::parse_peer("mad-lab-2026:9000", &host, &port) && host == "mad-lab-2026" && port == 9000,
          "failed to parse a hostname");
    CHECK(!pipe_tp_comm::parse_peer("no-port", &host, &port),     "accepted an address with no port");
    CHECK(!pipe_tp_comm::parse_peer("host:0", &host, &port),      "accepted port 0");
    CHECK(!pipe_tp_comm::parse_peer("host:99999", &host, &port),  "accepted an out-of-range port");
    CHECK(!pipe_tp_comm::parse_peer("host:80x", &host, &port),    "accepted a trailing garbage port");
}

int main() {
    printf("test-tp-comm\n");
    test_frame_header_layout();
    test_parse_peer();
    test_fixed_order_add_is_bit_identical();
    test_loopback_exchange();

    if (g_failures != 0) {
        fprintf(stderr, "test-tp-comm: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-tp-comm: OK\n");
    return 0;
}
