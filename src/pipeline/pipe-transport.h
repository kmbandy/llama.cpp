#pragma once

// Phase 2 cross-machine pipeline: blocking-TCP socket plumbing.
//
// Lifted from ggml/src/ggml-rpc/transport.cpp (socket_t create/connect/listen
// and the send_data/recv_data loop-until-complete helpers), with the RPC
// machinery stripped out: no remote op dispatch, no tensor addressing, no
// ggml types, no RDMA. This is byte transport only. The pipeline protocol
// (frame encode/decode) lives in pipe-protocol.*.
//
// Short reads and short writes loop to completion: send_data/recv_data return
// true only after every requested byte crossed. A short transfer is a
// transport failure (false), never a silently truncated buffer.

#include <cstddef>
#include <cstdint>
#include <memory>

struct pipe_socket_t;
typedef std::shared_ptr<pipe_socket_t> pipe_socket_ptr;

struct pipe_socket_t {
    ~pipe_socket_t();

    // Send/recv exactly `size` bytes, looping on short transfers. Returns
    // false on error or orderly peer shutdown (recv of 0). On false the
    // connection must be considered broken; no partial state is usable.
    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);

    // Interrupt blocking I/O without closing the descriptor.
    void shutdown();

    // Underlying descriptor, for callers that want to poll()/select() before
    // committing to a blocking recv_data. Returns -1 if unavailable. Added so
    // the expert worker can do useful work while waiting for the next request
    // instead of leaving its GPU idle -- on some cards (RX 480 / Polaris) the
    // cost of a submit depends on how long the GPU idled beforehand.
    // Do NOT read or write the socket through this; use send_data/recv_data.
    int poll_fd() const;

    // Accept one pending connection on a server socket. Returns nullptr on
    // failure. The returned socket has TCP_NODELAY set.
    pipe_socket_ptr accept();

    static pipe_socket_ptr create_server(const char * host, int port);
    static pipe_socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit pipe_socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

// No-op on POSIX; initialises Winsock on Windows. Call once before any
// pipe_socket_t use. Safe to call repeatedly.
bool pipe_transport_init();
void pipe_transport_shutdown();
