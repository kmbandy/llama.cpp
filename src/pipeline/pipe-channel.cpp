#include "pipe-channel.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <winsock2.h>
#else
#  include <cerrno>
#  include <poll.h>
#endif

namespace pipe_channel {
namespace {

using channel_clock = std::chrono::steady_clock;

int connect_retry_seconds() {
    const char * value = std::getenv("WP_DISPATCH_CONNECT_RETRY_S");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    const long seconds = std::strtol(value, nullptr, 10);
    return seconds > 0 ? (int) std::min(seconds, (long) INT_MAX) : 0;
}

pipe_socket_ptr connect_with_retry(const endpoint & target) {
    if (target.host.empty() || target.port <= 0 || target.port > 65535) {
        throw std::invalid_argument("pipe channel has an invalid endpoint");
    }
    const int retry_seconds = connect_retry_seconds();
    const auto deadline = channel_clock::now() + std::chrono::seconds(retry_seconds);
    bool retryable = false;
    pipe_socket_ptr socket;
    do {
        socket = pipe_socket_t::connect(target.host.c_str(), target.port, &retryable);
        if (socket || !retryable || retry_seconds == 0 || channel_clock::now() >= deadline) {
            break;
        }
        const auto delay = std::chrono::duration_cast<channel_clock::duration>(std::chrono::seconds(2));
        std::this_thread::sleep_for(std::min(delay, deadline - channel_clock::now()));
    } while (true);
    if (!socket) {
        throw std::runtime_error("pipe channel failed to connect to " + target.host + ":" +
                                 std::to_string(target.port));
    }
    return socket;
}

} // namespace

struct channel::impl {
    struct queued_frame {
        pipe_frame_type      type = PIPE_ERROR;
        uint64_t             seq_id = 0;
        std::vector<uint8_t> payload;
    };

    pipe_socket_ptr socket;
    std::string     peer;
    std::thread     writer;
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::deque<queued_frame> queue;
    uint64_t        next_seq_id = 1;
    bool            sending = false;
    bool            stop = false;
    bool            done = false;
    bool            failed = false;
    std::string     error;

    static constexpr size_t MAX_QUEUE = 8;

    impl(pipe_socket_ptr socket, std::string peer) : socket(std::move(socket)), peer(std::move(peer)) {
        if (!this->socket) {
            throw std::invalid_argument("pipe channel requires a socket");
        }
        writer = std::thread([this]() { writer_loop(); });
    }

    ~impl() {
        stop_writer();
    }

    void throw_if_failed_locked() const {
        if (failed) {
            throw std::runtime_error("pipe channel writer failed for " + peer + ": " + error);
        }
    }

    void writer_loop() noexcept {
        while (true) {
            queued_frame frame;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this]() { return stop || !queue.empty(); });
                if (queue.empty()) {
                    done = true;
                    cv.notify_all();
                    return;
                }
                frame = std::move(queue.front());
                queue.pop_front();
                sending = true;
            }

            const bool sent = pipe_send_frame(*socket, frame.type, frame.seq_id,
                                              frame.payload.data(), frame.payload.size());
            {
                std::lock_guard<std::mutex> lock(mutex);
                sending = false;
                if (!sent) {
                    failed = true;
                    error = "send failed";
                    queue.clear();
                    stop = true;
                    done = true;
                    if (socket) {
                        socket->shutdown();
                    }
                    cv.notify_all();
                    return;
                }
            }
            cv.notify_all();
        }
    }

    void stop_writer() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();

        const auto deadline = channel_clock::now() + std::chrono::seconds(3);
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait_until(lock, deadline, [this]() { return done; });
            if (!done && socket) {
                socket->shutdown();
            }
        }
        if (writer.joinable()) {
            writer.join();
        }
    }

    uint64_t send_request(pipe_frame_type type, std::vector<uint8_t> payload) {
        std::lock_guard<std::mutex> lock(mutex);
        throw_if_failed_locked();
        if (stop || next_seq_id == 0 || next_seq_id == UINT64_MAX) {
            throw std::runtime_error("pipe channel request sequence space is exhausted");
        }
        const uint64_t seq_id = next_seq_id++;
        enqueue_locked(type, seq_id, std::move(payload));
        return seq_id;
    }

    void send_frame(pipe_frame_type type, uint64_t seq_id, std::vector<uint8_t> payload) {
        std::lock_guard<std::mutex> lock(mutex);
        throw_if_failed_locked();
        if (stop) {
            throw std::runtime_error("pipe channel is closed");
        }
        enqueue_locked(type, seq_id, std::move(payload));
    }

    void enqueue_locked(pipe_frame_type type, uint64_t seq_id, std::vector<uint8_t> payload) {
        if (queue.size() >= MAX_QUEUE) {
            throw std::runtime_error("pipe channel writer queue is full for " + peer);
        }
        queue.push_back({ type, seq_id, std::move(payload) });
        cv.notify_all();
    }

    void flush() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this]() { return failed || (queue.empty() && !sending); });
        throw_if_failed_locked();
    }

    bool is_failed() const {
        std::lock_guard<std::mutex> lock(mutex);
        return failed;
    }
};

channel::channel(endpoint target) {
    if (!pipe_transport_init()) {
        throw std::runtime_error("pipe channel failed to initialize TCP transport");
    }
    const std::string peer = target.host + ":" + std::to_string(target.port);
    pimpl.reset(new impl(connect_with_retry(target), peer));
}

channel::channel(pipe_socket_ptr socket, std::string peer_name) {
    if (!pipe_transport_init()) {
        throw std::runtime_error("pipe channel failed to initialize TCP transport");
    }
    pimpl.reset(new impl(std::move(socket), std::move(peer_name)));
}

channel::~channel() = default;
channel::channel(channel &&) noexcept = default;
channel & channel::operator=(channel &&) noexcept = default;

uint64_t channel::send_request(pipe_frame_type type, std::vector<uint8_t> payload) {
    return pimpl->send_request(type, std::move(payload));
}

void channel::send_frame(pipe_frame_type type, uint64_t seq_id, std::vector<uint8_t> payload) {
    pimpl->send_frame(type, seq_id, std::move(payload));
}

void channel::flush() {
    pimpl->flush();
}

int channel::poll_fd() const {
    return pimpl->socket ? pimpl->socket->poll_fd() : -1;
}

const std::string & channel::peer_name() const {
    return pimpl->peer;
}

bool channel::harvest(const std::vector<channel *> & channels, received_frame & out, int timeout_ms) {
    if (timeout_ms < -1) {
        throw std::invalid_argument("pipe channel poll timeout is invalid");
    }
#ifdef _WIN32
    std::vector<WSAPOLLFD> pollfds;
#else
    std::vector<pollfd> pollfds;
#endif
    std::vector<channel *> ready_channels;
    pollfds.reserve(channels.size());
    ready_channels.reserve(channels.size());
    for (channel * value : channels) {
        if (value == nullptr || value->pimpl == nullptr || value->pimpl->is_failed()) {
            continue;
        }
        const int fd = value->poll_fd();
        if (fd < 0) {
            continue;
        }
#ifdef _WIN32
        pollfds.push_back({ (SOCKET) fd, POLLRDNORM, 0 });
#else
        pollfds.push_back({ fd, POLLIN, 0 });
#endif
        ready_channels.push_back(value);
    }
    if (pollfds.empty()) {
        return false;
    }

#ifdef _WIN32
    const int polled = WSAPoll(pollfds.data(), (ULONG) pollfds.size(), timeout_ms);
#else
    const int polled = poll(pollfds.data(), pollfds.size(), timeout_ms);
#endif
    if (polled == 0) {
        return false;
    }
    if (polled < 0) {
#ifndef _WIN32
        if (errno == EINTR) {
            return false;
        }
#endif
        throw std::runtime_error("pipe channel poll failed");
    }

    for (size_t i = 0; i < pollfds.size(); ++i) {
#ifdef _WIN32
        const short readable = POLLRDNORM;
#else
        const short readable = POLLIN;
#endif
        if ((pollfds[i].revents & (readable | POLLERR | POLLHUP | POLLNVAL)) == 0) {
            continue;
        }
        pipe_frame_type type;
        uint64_t seq_id = 0;
        std::vector<uint8_t> payload;
        if (!pipe_recv_frame(*ready_channels[i]->pimpl->socket, type, seq_id, payload)) {
            throw std::runtime_error("pipe channel peer closed " + ready_channels[i]->peer_name());
        }
        out.source = ready_channels[i];
        out.type = type;
        out.seq_id = seq_id;
        out.payload = std::move(payload);
        return true;
    }
    return false;
}

} // namespace pipe_channel
