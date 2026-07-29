#include "pipe-transport.h"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#else
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif

#include <mutex>

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

// Per-send/recv chunk cap. The RPC transport used 1 GiB; the pipeline moves
// ubatches, not tensors, so a much smaller cap keeps memory bounded while
// staying far above any single FWD_REQ payload (n_ubatch * n_embd * 4).
static constexpr size_t PIPE_MAX_CHUNK_SIZE = 64ull * 1024ull * 1024ull; // 64 MiB

#define PIPE_LOG_ERROR(...) std::fprintf(stderr, __VA_ARGS__)

struct pipe_socket_t::impl {
    explicit impl(sockfd_t fd) : fd(fd) {}
    ~impl();
    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);

    sockfd_t fd;
};

pipe_socket_t::impl::~impl() {
#ifdef _WIN32
    if (fd != INVALID_SOCKET) closesocket(fd);
#else
    if (fd >= 0) close(fd);
#endif
}

bool pipe_socket_t::impl::send_data(const void * data, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        size_t size_to_send = size - bytes_sent;
        if (size_to_send > PIPE_MAX_CHUNK_SIZE) {
            size_to_send = PIPE_MAX_CHUNK_SIZE;
        }
        ssize_t n = send(fd, (const char *) data + bytes_sent, size_to_send, 0);
        if (n < 0) {
            PIPE_LOG_ERROR("pipe send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            return false;
        }
        bytes_sent += (size_t) n;
    }
    return true;
}

bool pipe_socket_t::impl::recv_data(void * data, size_t size) {
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        size_t size_to_recv = size - bytes_recv;
        if (size_to_recv > PIPE_MAX_CHUNK_SIZE) {
            size_to_recv = PIPE_MAX_CHUNK_SIZE;
        }
        ssize_t n = recv(fd, (char *) data + bytes_recv, size_to_recv, 0);
        if (n < 0) {
            PIPE_LOG_ERROR("pipe recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            return false;
        }
        if (n == 0) {
            // orderly peer shutdown mid-stream: the connection is broken
            return false;
        }
        bytes_recv += (size_t) n;
    }
    return true;
}

/////////////////////////////////////////////////////////////////////////////

pipe_socket_t::pipe_socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

pipe_socket_t::~pipe_socket_t() = default;

bool pipe_socket_t::send_data(const void * data, size_t size) {
    return pimpl->send_data(data, size);
}

bool pipe_socket_t::recv_data(void * data, size_t size) {
    return pimpl->recv_data(data, size);
}

static bool is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    return sockfd != INVALID_SOCKET;
#else
    return sockfd >= 0;
#endif
}

static bool set_no_delay(sockfd_t sockfd) {
    int flag = 1;
    // TCP_NODELAY: hidden states and token ids are latency-sensitive, not
    // throughput-bound; disable Nagle.
    int ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));
    return ret == 0;
}

static bool set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *) &flag, sizeof(int));
    return ret == 0;
}

pipe_socket_ptr pipe_socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, NULL, NULL);
    if (!is_valid_fd(client_socket_fd)) {
        return nullptr;
    }
    if (!set_no_delay(client_socket_fd)) {
        PIPE_LOG_ERROR("pipe: failed to set TCP_NODELAY on accepted socket\n");
        return nullptr;
    }
    return pipe_socket_ptr(new pipe_socket_t(std::make_unique<impl>(client_socket_fd)));
}

pipe_socket_ptr pipe_socket_t::create_server(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) {
        return nullptr;
    }
    if (!set_reuse_addr(sockfd)) {
        PIPE_LOG_ERROR("pipe: failed to set SO_REUSEADDR\n");
        return nullptr;
    }
    if (inet_addr(host) == INADDR_NONE) {
        PIPE_LOG_ERROR("pipe: invalid host address: %s\n", host);
        return nullptr;
    }
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        PIPE_LOG_ERROR("pipe: bind failed on %s:%d\n", host, port);
        return nullptr;
    }
    if (listen(sockfd, 1) < 0) {
        PIPE_LOG_ERROR("pipe: listen failed on %s:%d\n", host, port);
        return nullptr;
    }
    return pipe_socket_ptr(new pipe_socket_t(std::make_unique<impl>(sockfd)));
}

pipe_socket_ptr pipe_socket_t::connect(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) {
        return nullptr;
    }
    if (!set_no_delay(sockfd)) {
        PIPE_LOG_ERROR("pipe: failed to set TCP_NODELAY\n");
        return nullptr;
    }
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    struct hostent * server = gethostbyname(host);
    if (server == NULL) {
        PIPE_LOG_ERROR("pipe: cannot resolve host '%s'\n", host);
        return nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    if (::connect(sockfd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        PIPE_LOG_ERROR("pipe: connect to %s:%d failed\n", host, port);
        return nullptr;
    }
    return pipe_socket_ptr(new pipe_socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
static std::mutex g_pipe_transport_mu;
static bool       g_pipe_transport_wsa_started = false;
#endif

bool pipe_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_pipe_transport_mu);
    if (g_pipe_transport_wsa_started) {
        return true;
    }
    WSADATA wsaData;
    int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (res != 0) {
        return false;
    }
    g_pipe_transport_wsa_started = true;
    return true;
#else
    return true;
#endif
}

void pipe_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_pipe_transport_mu);
    if (!g_pipe_transport_wsa_started) {
        return;
    }
    WSACleanup();
    g_pipe_transport_wsa_started = false;
#endif
}
