#include "pipe-tp-comm.h"
#include "pipe-transport.h"

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
}

pipe_tp_comm::~pipe_tp_comm() {
    if (print_stats_at_exit_) {
        print_stats("tp");
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
    pipe_socket_ptr server = pipe_socket_t::create_server(host.c_str(), port);
    if (!server) {
        return nullptr;
    }
    const uint64_t deadline = pipe_tp_now_ns() + (uint64_t) (timeout_ms > 0 ? timeout_ms : 0) * 1000000ull;
    for (;;) {
        pipe_socket_ptr peer = server->accept();
        if (peer) {
            std::unique_ptr<pipe_tp_comm> ret(new pipe_tp_comm(peer, max_values));
            if (ret->fd_ < 0 || !pipe_tp_set_nonblocking(ret->fd_)) {
                return nullptr;
            }
            return ret;
        }
        if (timeout_ms <= 0 || pipe_tp_now_ns() >= deadline) {
            return nullptr;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

std::unique_ptr<pipe_tp_comm> pipe_tp_comm::connect(
        const std::string & host, int port, size_t max_values, int timeout_ms) {
    if (!pipe_transport_init()) {
        return nullptr;
    }
    const uint64_t deadline = pipe_tp_now_ns() + (uint64_t) (timeout_ms > 0 ? timeout_ms : 0) * 1000000ull;
    for (;;) {
        bool retryable = false;
        pipe_socket_ptr sock = pipe_socket_t::connect(host.c_str(), port, &retryable);
        if (sock) {
            std::unique_ptr<pipe_tp_comm> ret(new pipe_tp_comm(sock, max_values));
            if (ret->fd_ < 0 || !pipe_tp_set_nonblocking(ret->fd_)) {
                return nullptr;
            }
            return ret;
        }
        // Never retry a protocol, DNS or local setup error; only a refusal or timeout.
        if (!retryable || timeout_ms <= 0 || pipe_tp_now_ns() >= deadline) {
            return nullptr;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
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
        const int pr = poll(&pfd, 1, -1);
        if (pr < 0) {
            if (errno == EINTR) {
                continue;
            }
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
