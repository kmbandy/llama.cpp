#include "pipe-tp-comm.h"
#include "pipe-transport.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#ifdef _WIN32
#   include <winsock2.h>
#else
#   include <errno.h>
#   include <fcntl.h>
#   include <poll.h>
#   include <sys/socket.h>
#   include <sys/uio.h>
#   include <unistd.h>
#endif

#ifndef MSG_NOSIGNAL
#   define MSG_NOSIGNAL 0
#endif

// ---------------------------------------------------------------------------------------------
// the determinism rule
// ---------------------------------------------------------------------------------------------

void pipe_tp_add_fixed_order(float * local, const float * peer, size_t n_values, bool local_is_rank0) {
    // S_total = S_rank0 + S_rank1, in that operand order, on both ranks. Written out as two loops
    // rather than one so the invariant is visible in the source and cannot be lost to a later
    // refactor; the two are bit-identical because IEEE-754 binary32 addition of two finite values
    // is commutative and correctly rounded.
    if (local_is_rank0) {
        for (size_t i = 0; i < n_values; i++) {
            local[i] = local[i] + peer[i]; // S0 + S1
        }
    } else {
        for (size_t i = 0; i < n_values; i++) {
            local[i] = peer[i] + local[i]; // S0 + S1
        }
    }
}

// ---------------------------------------------------------------------------------------------

static uint64_t pipe_tp_now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool pipe_tp_set_nonblocking(int fd) {
#ifdef _WIN32
    u_long mode = 1;
    return ioctlsocket((SOCKET) fd, FIONBIO, &mode) == 0;
#else
    const int flags = fcntl(fd, F_GETFL, 0);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
#endif
}

pipe_tp_comm::pipe_tp_comm(std::shared_ptr<pipe_socket_t> sock, size_t max_values)
        : sock_(std::move(sock)), max_values_(max_values) {
    fd_ = sock_->poll_fd();
    // Allocated once, never zeroed: every byte is overwritten by the peer's payload before it is
    // read. resize()/assign() on a vector would cost a max-width memset per construction and, in
    // the anti-pattern this file exists to avoid, per frame.
    recv_buf_.reset(new float[max_values]);
    const char * env = getenv("WP_TP_STATS");
    print_stats_at_exit_ = env != nullptr && atoi(env) != 0;
    // A peer that has DIED is detected by the socket (POLLHUP / recv 0) and needs no timeout. A
    // peer that is merely STUCK - a follower wedged in a graph, a rank waiting for a frame the
    // other never sends because the graphs diverged - would otherwise hang this rank forever
    // inside poll(-1). Default 120 s: far longer than any single ubatch on this rig, short enough
    // that a wedged run fails a request instead of a shift.
    const char * env_to = getenv("WP_TP_TIMEOUT_MS");
    timeout_ms_ = env_to != nullptr ? atoi(env_to) : 120000;
    if (timeout_ms_ < 0) {
        timeout_ms_ = 0;
    }
}

pipe_tp_comm::~pipe_tp_comm() {
    if (print_stats_at_exit_) {
        print_stats("tp");
    }
}

// ---------------------------------------------------------------------------------------------
// establishing the connection
//
// THE ORDERING BUG THIS SOLVES. The peer socket used to be created inside llama_context, i.e.
// AFTER the model load. The two ranks' loads are not the same length - on this rig the leader
// reads BF16 weights off a spinning disk for ~7 minutes while the follower comes up in ~43
// seconds - so the follower's connect window opened and closed long before the leader had bound
// the port at all, and the follower died with "failed to establish the peer connection".
//
// Binding EARLY fixes it from both directions at once. Once listen() has been called the kernel
// completes the TCP handshake from the backlog on its own, so the follower's connect() succeeds
// the moment the port exists, whether or not the leader has reached its accept() - which means
// the leader may still be loading, and start order stops mattering.
// ---------------------------------------------------------------------------------------------

// The listening socket, opened by prebind() long before any model is touched and picked up by the
// matching listen() call later. Process-global because the two calls are minutes and several
// stack frames apart; there is exactly one tensor-parallel world per process.
static pipe_socket_ptr g_tp_prebound;
static std::string     g_tp_prebound_host;
static int             g_tp_prebound_port = 0;

int pipe_tp_comm::default_connect_timeout_ms() {
    const char * env = getenv("WP_TP_CONNECT_TIMEOUT_MS");
    if (env != nullptr) {
        const int v = atoi(env);
        return v < 0 ? 0 : v;
    }
    return 30 * 60 * 1000; // 30 minutes
}

bool pipe_tp_comm::prebind(const std::string & host, int port) {
    if (!pipe_transport_init()) {
        return false;
    }
    if (g_tp_prebound && g_tp_prebound_host == host && g_tp_prebound_port == port) {
        return true; // idempotent
    }
    if (g_tp_prebound) {
        fprintf(stderr, "pipe-tp: refusing to rebind: already listening on %s:%d\n",
                g_tp_prebound_host.c_str(), g_tp_prebound_port);
        return false;
    }
    pipe_socket_ptr server = pipe_socket_t::create_server(host.c_str(), port);
    if (!server) {
        fprintf(stderr, "pipe-tp: failed to bind %s:%d (note: the host must be a dotted-quad, "
                        "e.g. 0.0.0.0, not a name)\n", host.c_str(), port);
        return false;
    }
    g_tp_prebound      = server;
    g_tp_prebound_host = host;
    g_tp_prebound_port = port;
    return true;
}

// Report progress every ten seconds while waiting, so that a wait is visibly a wait. A silent
// process is indistinguishable from a hung one, and both of the failures this code path has
// actually produced looked like hangs.
static void pipe_tp_waiting_note(const char * what, const std::string & host, int port,
                                 uint64_t t0_ns, uint64_t * next_note_ns, int timeout_ms) {
    const uint64_t now = pipe_tp_now_ns();
    if (now < *next_note_ns) {
        return;
    }
    *next_note_ns = now + 10ull * 1000000000ull;
    if (timeout_ms > 0) {
        fprintf(stderr, "pipe-tp: %s %s:%d ... %.0f s elapsed of at most %.0f s\n",
                what, host.c_str(), port, (now - t0_ns) / 1e9, timeout_ms / 1000.0);
    } else {
        fprintf(stderr, "pipe-tp: %s %s:%d ... %.0f s elapsed (no timeout)\n",
                what, host.c_str(), port, (now - t0_ns) / 1e9);
    }
}

bool pipe_tp_comm::parse_peer(const std::string & spec, std::string * host, int * port) {
    const size_t colon = spec.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= spec.size()) {
        return false;
    }
    *host = spec.substr(0, colon);
    char * end = nullptr;
    const long p = strtol(spec.c_str() + colon + 1, &end, 10);
    if (end == nullptr || *end != '\0' || p <= 0 || p > 65535) {
        return false;
    }
    *port = (int) p;
    return true;
}

std::unique_ptr<pipe_tp_comm> pipe_tp_comm::listen(
        const std::string & host, int port, size_t max_values, int timeout_ms) {
    if (!pipe_transport_init()) {
        return nullptr;
    }

    // Normally the socket is already open: prebind() bound it before the model load. Falling back
    // to binding here keeps the call self-contained for the tests and for any caller that has not
    // been taught to prebind, but on the real path this branch should not be taken - if it is, the
    // port only became available after the load and the follower may already have given up.
    pipe_socket_ptr server;
    if (g_tp_prebound && g_tp_prebound_host == host && g_tp_prebound_port == port) {
        server = g_tp_prebound;
    } else {
        server = pipe_socket_t::create_server(host.c_str(), port);
        if (!server) {
            return nullptr;
        }
    }

    const int listen_fd = server->poll_fd();
    if (listen_fd < 0) {
        return nullptr;
    }

    const uint64_t t0       = pipe_tp_now_ns();
    const uint64_t deadline = t0 + (uint64_t) (timeout_ms > 0 ? timeout_ms : 0) * 1000000ull;
    uint64_t next_note = t0 + 10ull * 1000000000ull;

    for (;;) {
#ifndef _WIN32
        // poll the LISTENING fd rather than calling the blocking accept() directly. accept() on a
        // blocking socket never returns until a peer arrives, which made both the timeout below
        // and any progress logging dead code - a leader waiting for a follower that had already
        // exited simply sat there silently, forever, which is exactly what it was observed doing.
        struct pollfd pfd;
        pfd.fd      = listen_fd;
        pfd.events  = POLLIN;
        pfd.revents = 0;
        const int pr = poll(&pfd, 1, 1000);
        if (pr < 0 && errno != EINTR) {
            return nullptr;
        }
        if (pr > 0 && (pfd.revents & POLLIN)) {
            pipe_socket_ptr peer = server->accept();
            if (peer) {
                std::unique_ptr<pipe_tp_comm> ret(new pipe_tp_comm(peer, max_values));
                if (ret->fd_ < 0 || !pipe_tp_set_nonblocking(ret->fd_)) {
                    return nullptr;
                }
                fprintf(stderr, "pipe-tp: peer rank connected on %s:%d after %.1f s\n",
                        host.c_str(), port, (pipe_tp_now_ns() - t0) / 1e9);
                return ret;
            }
        }
#else
        pipe_socket_ptr peer = server->accept();
        if (peer) {
            std::unique_ptr<pipe_tp_comm> ret(new pipe_tp_comm(peer, max_values));
            if (ret->fd_ < 0 || !pipe_tp_set_nonblocking(ret->fd_)) {
                return nullptr;
            }
            return ret;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
#endif
        pipe_tp_waiting_note("waiting for the peer rank to connect on", host, port, t0, &next_note, timeout_ms);
        if (timeout_ms > 0 && pipe_tp_now_ns() >= deadline) {
            fprintf(stderr, "pipe-tp: no peer rank connected on %s:%d within %.0f s\n",
                    host.c_str(), port, timeout_ms / 1000.0);
            return nullptr;
        }
    }
}

std::unique_ptr<pipe_tp_comm> pipe_tp_comm::connect(
        const std::string & host, int port, size_t max_values, int timeout_ms) {
    if (!pipe_transport_init()) {
        return nullptr;
    }
    const uint64_t t0       = pipe_tp_now_ns();
    const uint64_t deadline = t0 + (uint64_t) (timeout_ms > 0 ? timeout_ms : 0) * 1000000ull;
    uint64_t next_note   = t0 + 10ull * 1000000000ull;
    // Modest backoff, capped at five seconds. The cap is set by NOISE, not by latency: every
    // failed attempt makes pipe_socket_t::connect print a line, and against a seven-minute leader
    // load a five-second reconnect granularity is free while a 50 ms one would bury the progress
    // notes under twenty thousand identical failures.
    uint64_t backoff_ms  = 50;
    bool     first_try   = true;

    for (;;) {
        bool retryable = false;
        pipe_socket_ptr sock = pipe_socket_t::connect(host.c_str(), port, &retryable);
        if (sock) {
            std::unique_ptr<pipe_tp_comm> ret(new pipe_tp_comm(sock, max_values));
            if (ret->fd_ < 0 || !pipe_tp_set_nonblocking(ret->fd_)) {
                return nullptr;
            }
            fprintf(stderr, "pipe-tp: connected to the leader rank at %s:%d after %.1f s\n",
                    host.c_str(), port, (pipe_tp_now_ns() - t0) / 1e9);
            return ret;
        }

        // A name that does not resolve, or a local setup failure, is deterministic: it will fail
        // identically for the next thirty minutes, so fail it now. But only on the FIRST attempt -
        // once we know the address is usable, a later non-retryable error is a transient network
        // condition (an interface flapping, a host rebooting) and is worth waiting out, which is
        // the entire point of a thirty-minute cap.
        if (first_try && !retryable) {
            fprintf(stderr, "pipe-tp: cannot reach %s:%d and the failure is not transient "
                            "(unresolvable host, or a local socket error)\n", host.c_str(), port);
            return nullptr;
        }
        first_try = false;

        if (timeout_ms > 0 && pipe_tp_now_ns() >= deadline) {
            fprintf(stderr, "pipe-tp: gave up connecting to the leader rank at %s:%d after %.0f s. "
                            "If the leader is still loading its model, raise "
                            "WP_TP_CONNECT_TIMEOUT_MS.\n", host.c_str(), port, timeout_ms / 1000.0);
            return nullptr;
        }
        pipe_tp_waiting_note("waiting for the leader rank at", host, port, t0, &next_note, timeout_ms);

        std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        backoff_ms = backoff_ms < 5000 ? std::min<uint64_t>(backoff_ms * 2, 5000) : 5000;
    }
}

bool pipe_tp_comm::exchange_add(float * local, size_t n_values, bool local_is_rank0) {
    if (fd_ < 0 || n_values == 0) {
        return false;
    }
    if (n_values > max_values_) {
        // The caller sizes the buffer for the widest partial it expects; a wider one is not a
        // reason to fail the graph. Grow once and stay allocation-free from then on.
        fprintf(stderr, "pipe-tp: growing receive buffer %zu -> %zu values\n", max_values_, n_values);
        recv_buf_.reset(new float[n_values]);
        max_values_ = n_values;
    }
    const uint64_t t0 = pipe_tp_now_ns();

    const size_t payload_bytes = n_values * sizeof(float);

    pipe_tp_frame_hdr hdr_out;
    hdr_out.magic         = PIPE_TP_MAGIC;
    hdr_out.type          = PIPE_TP_MSG_REDUCE;
    hdr_out.dtype         = (uint8_t) dtype_;
    hdr_out.flags         = 0;
    hdr_out.seq           = ++seq_out_;
    hdr_out.payload_bytes = (uint32_t) payload_bytes;

    pipe_tp_frame_hdr hdr_in;
    memset(&hdr_in, 0, sizeof(hdr_in));

#ifndef _WIN32
    // ONE gathered write: header immediately followed by the caller's payload, so the header never
    // leaves as its own TCP segment. iov is advanced in place on a short write.
    struct iovec iov[2];
    iov[0].iov_base = &hdr_out;
    iov[0].iov_len  = sizeof(hdr_out);
    iov[1].iov_base = local;
    iov[1].iov_len  = payload_bytes;
    int    iov_i    = 0;
    size_t to_send  = sizeof(hdr_out) + payload_bytes;

    uint8_t * recv_dst_hdr = (uint8_t *) &hdr_in;
    size_t recv_hdr_done  = 0;
    size_t recv_body_done = 0;
    bool   hdr_validated  = false;
    size_t recv_body_todo = payload_bytes; // provisional; confirmed against the header below

    while (to_send > 0 || recv_hdr_done < sizeof(hdr_in) || recv_body_done < recv_body_todo) {
        struct pollfd pfd;
        pfd.fd      = fd_;
        pfd.events  = 0;
        pfd.revents = 0;
        if (to_send > 0) {
            pfd.events |= POLLOUT;
        }
        if (recv_hdr_done < sizeof(hdr_in) || recv_body_done < recv_body_todo) {
            pfd.events |= POLLIN;
        }
        const int pr = poll(&pfd, 1, timeout_ms_ > 0 ? timeout_ms_ : -1);
        if (pr < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (pr == 0) {
            fprintf(stderr, "pipe-tp: peer silent for %d ms during reduce %u of %zu values - "
                            "treating the connection as dead rather than hanging the graph\n",
                    timeout_ms_, seq_out_, n_values);
            return false;
        }
        if (pfd.revents & (POLLERR | POLLNVAL)) {
            return false;
        }

        if ((pfd.revents & POLLOUT) && to_send > 0) {
            struct msghdr msg;
            memset(&msg, 0, sizeof(msg));
            msg.msg_iov    = &iov[iov_i];
            msg.msg_iovlen = 2 - iov_i;
            const ssize_t n = sendmsg(fd_, &msg, MSG_NOSIGNAL);
            if (n < 0) {
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                    // fall through to the receive side; poll again next iteration
                } else {
                    return false;
                }
            } else {
                to_send -= (size_t) n;
                size_t left = (size_t) n;
                while (left > 0 && iov_i < 2) {
                    if (left >= iov[iov_i].iov_len) {
                        left -= iov[iov_i].iov_len;
                        iov[iov_i].iov_len = 0;
                        iov_i++;
                    } else {
                        iov[iov_i].iov_base = (char *) iov[iov_i].iov_base + left;
                        iov[iov_i].iov_len -= left;
                        left = 0;
                    }
                }
                stats_.bytes_sent += (uint64_t) n;
            }
        }

        if (pfd.revents & (POLLIN | POLLHUP)) {
            if (recv_hdr_done < sizeof(hdr_in)) {
                const ssize_t n = recv(fd_, recv_dst_hdr + recv_hdr_done, sizeof(hdr_in) - recv_hdr_done, 0);
                if (n == 0) {
                    return false; // orderly peer shutdown mid-frame
                }
                if (n < 0) {
                    if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
                        return false;
                    }
                } else {
                    recv_hdr_done += (size_t) n;
                    stats_.bytes_recvd += (uint64_t) n;
                }
            }
            if (!hdr_validated && recv_hdr_done == sizeof(hdr_in)) {
                if (hdr_in.magic != PIPE_TP_MAGIC ||
                        hdr_in.type          != PIPE_TP_MSG_REDUCE ||
                        hdr_in.dtype         != (uint8_t) dtype_ ||
                        hdr_in.flags         != 0 ||
                        hdr_in.payload_bytes != (uint32_t) payload_bytes ||
                        hdr_in.seq           != seq_out_) {
                    // The two ranks are in lockstep: the peer's Nth reduce must be this rank's Nth
                    // reduce, of the same width. Anything else means the graphs diverged, which is
                    // a bug to surface immediately rather than a condition to recover from.
                    return false;
                }
                seq_in_        = hdr_in.seq;
                recv_body_todo = hdr_in.payload_bytes;
                hdr_validated  = true;
            }
            if (hdr_validated && recv_body_done < recv_body_todo) {
                const ssize_t n = recv(fd_, (uint8_t *) recv_buf_.get() + recv_body_done,
                                       recv_body_todo - recv_body_done, 0);
                if (n == 0) {
                    return false;
                }
                if (n < 0) {
                    if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
                        return false;
                    }
                } else {
                    recv_body_done += (size_t) n;
                    stats_.bytes_recvd += (uint64_t) n;
                }
            }
        }
    }
#else
    // Windows has no sendmsg; correctness only, this rig is Linux.
    (void) local_is_rank0;
    return false;
#endif

    pipe_tp_add_fixed_order(local, recv_buf_.get(), n_values, local_is_rank0);

    const uint64_t dt = pipe_tp_now_ns() - t0;
    stats_.n_exchanges++;
    stats_.ns_blocked += dt;
    if (dt > stats_.ns_blocked_max) {
        stats_.ns_blocked_max = dt;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// M3 control channel
// ---------------------------------------------------------------------------------------------

#ifndef _WIN32
// Drive one direction of a small transfer to completion, honouring timeout_ms_. `send` selects the
// direction. Returns false on a dead or silent peer.
static bool pipe_tp_xfer(int fd, uint8_t * buf, size_t bytes, bool send_dir, int timeout_ms) {
    size_t done = 0;
    while (done < bytes) {
        struct pollfd pfd;
        pfd.fd      = fd;
        pfd.events  = send_dir ? POLLOUT : POLLIN;
        pfd.revents = 0;
        const int pr = poll(&pfd, 1, timeout_ms > 0 ? timeout_ms : -1);
        if (pr < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (pr == 0) {
            return false; // silent peer
        }
        if (pfd.revents & (POLLERR | POLLNVAL)) {
            return false;
        }
        const ssize_t n = send_dir
            ? send(fd, buf + done, bytes - done, MSG_NOSIGNAL)
            : recv(fd, buf + done, bytes - done, 0);
        if (n == 0) {
            return false; // orderly shutdown mid-message
        }
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            return false;
        }
        done += (size_t) n;
    }
    return true;
}
#endif

bool pipe_tp_comm::send_msg(uint8_t type, const void * payload, size_t bytes) {
#ifdef _WIN32
    (void) type; (void) payload; (void) bytes;
    return false;
#else
    if (fd_ < 0 || type == PIPE_TP_MSG_REDUCE || bytes > 0xffffffffull) {
        return false;
    }
    pipe_tp_frame_hdr hdr;
    hdr.magic         = PIPE_TP_MAGIC;
    hdr.type          = type;
    hdr.dtype         = (uint8_t) dtype_;
    hdr.flags         = 0;
    hdr.seq           = ++msg_seq_out_;
    hdr.payload_bytes = (uint32_t) bytes;

    // Two writes, not a gathered one: unlike a reduce this is not on the per-layer hot path, and
    // the peer is blocked reading rather than simultaneously writing, so there is no deadlock to
    // avoid and no wakeup budget to protect.
    if (!pipe_tp_xfer(fd_, (uint8_t *) &hdr, sizeof(hdr), true, msg_timeout_ms_)) {
        return false;
    }
    if (bytes > 0 && !pipe_tp_xfer(fd_, (uint8_t *) payload, bytes, true, msg_timeout_ms_)) {
        return false;
    }
    stats_.bytes_sent += sizeof(hdr) + bytes;
    return true;
#endif
}

bool pipe_tp_comm::recv_msg(uint8_t * type, std::vector<uint8_t> & payload) {
#ifdef _WIN32
    (void) type; (void) payload;
    return false;
#else
    if (fd_ < 0) {
        return false;
    }
    pipe_tp_frame_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    if (!pipe_tp_xfer(fd_, (uint8_t *) &hdr, sizeof(hdr), false, msg_timeout_ms_)) {
        return false;
    }
    if (hdr.magic != PIPE_TP_MAGIC || hdr.flags != 0 || hdr.dtype != (uint8_t) dtype_) {
        fprintf(stderr, "pipe-tp: malformed control frame header (magic=%08x type=%u dtype=%u flags=%u)\n",
                hdr.magic, hdr.type, hdr.dtype, hdr.flags);
        return false;
    }
    if (hdr.type == PIPE_TP_MSG_REDUCE) {
        // A reduce frame where a descriptor was expected means the peer entered a graph this rank
        // has not been told about: the ranks have diverged. Never silently skip it.
        fprintf(stderr, "pipe-tp: received a REDUCE frame while waiting for a control frame - "
                        "the two ranks have diverged\n");
        return false;
    }
    if (hdr.seq != msg_seq_in_ + 1) {
        fprintf(stderr, "pipe-tp: control frame out of order (expected seq %u, got %u)\n",
                msg_seq_in_ + 1, hdr.seq);
        return false;
    }
    msg_seq_in_ = hdr.seq;

    payload.resize(hdr.payload_bytes);
    if (hdr.payload_bytes > 0 &&
            !pipe_tp_xfer(fd_, payload.data(), hdr.payload_bytes, false, msg_timeout_ms_)) {
        return false;
    }
    stats_.bytes_recvd += sizeof(hdr) + hdr.payload_bytes;
    *type = hdr.type;
    return true;
#endif
}

void pipe_tp_comm::print_stats(const char * tag) const {
    const double ms_total = stats_.ns_blocked / 1e6;
    fprintf(stderr,
        "%s: exchanges=%llu bytes_sent=%llu bytes_recvd=%llu ns_blocked=%llu (%.1f ms total, "
        "%.3f ms mean, %.3f ms max)\n",
        tag,
        (unsigned long long) stats_.n_exchanges,
        (unsigned long long) stats_.bytes_sent,
        (unsigned long long) stats_.bytes_recvd,
        (unsigned long long) stats_.ns_blocked,
        ms_total,
        stats_.n_exchanges ? ms_total / (double) stats_.n_exchanges : 0.0,
        stats_.ns_blocked_max / 1e6);
}
