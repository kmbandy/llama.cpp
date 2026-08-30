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

    // Send `a` immediately followed by `b` as ONE gathered write (sendmsg with
    // two iovecs on POSIX; two ordinary sends on Windows). Byte-for-byte
    // identical on the wire to send_data(a) followed by send_data(b) -- this
    // exists purely to stop a small frame header from leaving as its own TCP
    // segment. With TCP_NODELAY set (we set it on every socket) a separate
    // send() of the 32-byte header is pushed out immediately as a standalone
    // packet; the peer's recv_data(header) then returns on that packet and
    // blocks again for the body, costing an extra kernel wakeup per frame per
    // direction. Both of those wakeups sit inside the requester's wait and
    // outside the responder's own service clock. Short writes loop to
    // completion exactly as send_data does.
    bool send_data2(const void * a, size_t a_size, const void * b, size_t b_size);

    // True when the peer of this connection is on the loopback interface
    // (IPv4 127.0.0.0/8). Determined ONCE, from the peer address, at the
    // moment the connection is established: from the resolved destination in
    // connect(), and from accept()'s returned peer sockaddr on the server
    // side. Never re-queried, so it costs nothing on the hot path.
    //
    // This exists because the header/payload coalescing decision is not
    // uniform across the fabric: the gathered write measured as a LOSS on the
    // loopback leg and is untested on the 1 GbE leg, so WP_SEND_COALESCE=2
    // needs a per-socket answer to "is this the wire or the kernel?".
    //
    // CAVEAT for callers: this is a literal 127.0.0.0/8 test, not a
    // "does this traverse a NIC?" test. A connection to this host's OWN LAN
    // address (e.g. 192.168.1.33 talking to 192.168.1.33) is delivered
    // locally by the kernel but reports false here. Listening on a server
    // socket that never had a peer also reports false.
    bool peer_is_loopback() const;

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
    // When `retryable` is supplied, it is set only for a TCP refusal or
    // timeout.  Callers must not retry protocol, DNS, or local setup errors.
    static pipe_socket_ptr connect(const char * host, int port, bool * retryable = nullptr);

private:
    struct impl;
    explicit pipe_socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

// No-op on POSIX; initialises Winsock on Windows. Call once before any
// pipe_socket_t use. Safe to call repeatedly.
bool pipe_transport_init();
void pipe_transport_shutdown();
