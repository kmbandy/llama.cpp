#pragma once

// Cross-host tensor-parallel exchange.
//
// One persistent, full-duplex TCP connection between two ranks of a tensor-parallel world. At
// every reduce point of the meta backend's graph (two per transformer layer for qwen35, so 128
// per token for a 64-layer trunk) each rank has an n_embd x n_tokens F32 PARTIAL sum; this class
// ships that partial to the peer, receives the peer's, and adds them.
//
// DESIGN CONSTRAINTS, all of them load-bearing on a 1 GbE link carrying 128 exchanges per token:
//
//   - ONE syscall per direction per exchange. The 16-byte header and the payload leave as a
//     single gathered write (sendmsg with two iovecs). Sending the header separately with Nagle
//     off puts it on the wire as its own segment and costs the peer an extra kernel wakeup per
//     frame per direction - 256 extra wakeups per token, 8-15 ms at 30-60 us each, which is the
//     single largest avoidable cost in the design.
//
//   - NO ALLOCATION, NO ZERO-FILL, NO COPY on the hot path. The receive buffer is allocated once,
//     sized for the widest ubatch, and reused forever; the send side transmits the caller's buffer
//     in place, so a pinned host staging buffer owned by the backend never has to be copied.
//
//   - GENUINELY FULL DUPLEX. Both ranks send and receive the same frame at the same time. A
//     blocking send-then-recv would deadlock as soon as a payload exceeds the socket buffers
//     (10 MB at n_embd 5120 and ubatch 512), so the fd is non-blocking and one poll() loop drives
//     both directions until both complete.
//
// DETERMINISM. The sum is defined as
//
//     S_total = S_rank0 + S_rank1
//
// in that operand order, in F32, on BOTH ranks, regardless of which rank is executing. IEEE-754
// binary32 addition of two finite values is commutative and correctly rounded, so a+b == b+a
// bit-exactly and the rule is trivially satisfied for a two-rank world - it is written down, and
// tested, so that a third rank or a future fused form cannot silently break it.
//
// WIRE DTYPE. F32 today. The header carries a dtype byte so an F16 arm (worth ~1.6x on this link,
// more than speculation buys) can be added behind an env gate without a protocol change. F16 is
// NOT bit-identical to the single-process reference by construction and must be qualified on NLL,
// not on token identity.
//
// FRAME LAYOUT. The header is deliberately a superset of what the reduce path needs so that the
// M3 control channel (HELLO / DECODE / CTRL) can share this one connection: a type byte selects
// the message and a length says how far to skip, so a reader can route frames it does not handle.
//
// Both hosts of this rig are little-endian x86-64 and the header is sent as raw bytes; a
// big-endian peer would need byte-swapping that is deliberately not paid for here.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct pipe_socket_t;

// Message types on the TP connection. Only REDUCE is implemented here; the rest are reserved so
// that M3 can add them without renumbering.
enum pipe_tp_msg_type : uint8_t {
    PIPE_TP_MSG_INVALID = 0,
    PIPE_TP_MSG_REDUCE  = 1, // payload: the sender's partial sum, `payload_bytes` of `dtype`
    PIPE_TP_MSG_HELLO   = 2, // M3: model identity, world/rank, tensor_split, context shape
    PIPE_TP_MSG_DECODE  = 3, // M3: per-step batch descriptor (pipe_tp_encode_batch)
    PIPE_TP_MSG_CTRL    = 4, // M3: memory mutation / shutdown (pipe_tp_encode_ctrl)
    PIPE_TP_MSG_ERROR   = 5, // M3: a rank is aborting; payload is a UTF-8 reason
};

enum pipe_tp_wire_dtype : uint8_t {
    PIPE_TP_DTYPE_F32 = 0,
    PIPE_TP_DTYPE_F16 = 1, // reserved, env-gated
};

#define PIPE_TP_MAGIC 0x50545057u // "WPTP" little-endian

// 16 bytes, no implicit padding on any supported ABI.
struct pipe_tp_frame_hdr {
    uint32_t magic;
    uint8_t  type;
    uint8_t  dtype;
    uint16_t flags;         // reserved, must be 0
    uint32_t seq;           // monotonically increasing per direction, starts at 1
    uint32_t payload_bytes;
};
static_assert(sizeof(pipe_tp_frame_hdr) == 16, "pipe_tp_frame_hdr must be 16 bytes");

struct pipe_tp_stats {
    uint64_t n_exchanges   = 0;
    uint64_t bytes_sent    = 0;
    uint64_t bytes_recvd   = 0;
    uint64_t ns_blocked    = 0; // wall time inside exchange_add(), i.e. time the graph is stalled
    uint64_t ns_blocked_max = 0;
};

// Add a peer's partial into a local partial in the FIXED order S_rank0 + S_rank1.
//
// Exposed (and tested) separately from the transport so the determinism rule can be checked
// without a socket. `local_is_rank0` selects which operand is rank 0's; the two role assignments
// must produce bit-identical results.
void pipe_tp_add_fixed_order(float * local, const float * peer, size_t n_values, bool local_is_rank0);

struct pipe_tp_comm {
    ~pipe_tp_comm();

    // Bind and listen on `host:port` NOW - called long before the model is loaded, so that a rank
    // whose weights take minutes to come off disk is nevertheless REFUSING NOTHING from the moment
    // it starts. Without this, rank 1 can only connect after rank 0's load has finished, and the
    // two loads are not the same length: a 7-minute HDD load on the leader against a 43-second
    // load on the follower means the follower's connect window opens and closes before the leader
    // has bound the port at all. The listening socket is process-global and is picked up by the
    // matching listen() call later. Idempotent; safe to call when this rank is not rank 0 (it
    // simply should not be). `host` must be a dotted-quad (0.0.0.0 to accept on every interface),
    // not a name: the underlying create_server() resolves with inet_addr().
    static bool prebind(const std::string & host, int port);

    // Cap on establishing the peer connection, in milliseconds. WP_TP_CONNECT_TIMEOUT_MS, default
    // 30 minutes. It is long ON PURPOSE: the two ranks' model loads differ by minutes on this rig,
    // and whichever finishes first has nothing useful to do but wait. A cap that expires is a
    // configuration error being reported, not a deadline anybody wants enforced.
    static int default_connect_timeout_ms();

    // Rank 0 accepts one peer on the socket prebind() opened (or binds `host:port` itself if
    // prebind was not called); every other rank connects to it. Both block until the connection is
    // established or `timeout_ms` elapses (<= 0 means a single attempt), logging progress every
    // ten seconds so a wait is visibly a wait and not a hang.
    // `max_values` sizes the receive buffer up front; it should be at least n_embd * n_ubatch,
    // the widest partial normally exchanged. A wider partial grows the buffer once (with a line to
    // stderr) rather than failing, so the steady state stays allocation-free either way.
    static std::unique_ptr<pipe_tp_comm> listen (const std::string & host, int port, size_t max_values, int timeout_ms);
    static std::unique_ptr<pipe_tp_comm> connect(const std::string & host, int port, size_t max_values, int timeout_ms);

    // Parse "host:port". Returns false on a malformed address.
    static bool parse_peer(const std::string & spec, std::string * host, int * port);

    // ---------------------------------------------------------------------------------------
    // M3 control channel. These share the connection with the reduce frames above; ordering does
    // the separation, because the leader always finishes sending a descriptor before it enters
    // the graph that produces the reduce frames explaining it.
    //
    // They keep their OWN sequence counter. exchange_add() checks that the peer's Nth reduce is
    // this rank's Nth reduce, which is the whole lockstep guarantee of M2; control frames flow in
    // one direction only (leader -> follower), so counting them on the shared counter would make
    // the two ranks' reduce numbering drift apart by exactly the number of control frames sent.
    // ---------------------------------------------------------------------------------------

    // Send one non-reduce frame. Blocking; these are tens of bytes to a few KB, one per decode
    // step, so the poll loop of exchange_add() would be all cost and no benefit here.
    bool send_msg(uint8_t type, const void * payload, size_t bytes);

    // Receive one non-reduce frame. `payload` is resized to the frame's length. Returns false on
    // a broken connection, a malformed header, or a REDUCE frame arriving where a control frame
    // was expected (which means the ranks have diverged).
    bool recv_msg(uint8_t * type, std::vector<uint8_t> & payload);

    // Exchange one partial with the peer and leave the total in `local`.
    //
    // `local` holds this rank's partial on entry and S_rank0 + S_rank1 on return. It is sent in
    // place, so it may be (and should be) a pinned host buffer the backend reads the device
    // tensor into. Returns false if the connection broke or the peer sent an unexpected frame; the
    // connection must be considered dead after a false.
    bool exchange_add(float * local, size_t n_values, bool local_is_rank0);

    const pipe_tp_stats & stats() const { return stats_; }

    // One line to stderr. Called automatically from the destructor when WP_TP_STATS=1.
    void print_stats(const char * tag) const;

private:
    pipe_tp_comm(std::shared_ptr<pipe_socket_t> sock, size_t max_values);

    std::shared_ptr<pipe_socket_t>  sock_;
    int                             fd_ = -1;
    std::unique_ptr<float[]>        recv_buf_;
    size_t                          max_values_ = 0;
    uint32_t                        seq_out_    = 0;
    uint32_t                        seq_in_     = 0;
    uint32_t                        msg_seq_out_ = 0;
    uint32_t                        msg_seq_in_  = 0;
    // Two different clocks, deliberately.
    //   timeout_ms_     bounds a REDUCE. Both ranks are inside the same graph, so the peer's
    //                   partial is due within one ubatch's compute; silence past that is a wedge.
    //                   WP_TP_TIMEOUT_MS, default 120 s.
    //   msg_timeout_ms_ bounds a CONTROL frame, and is 0 (block forever) because its arrival time
    //                   is bounded by USER behaviour, not by compute: a follower attached to an
    //                   idle llama-server legitimately waits hours between requests, and during
    //                   startup it waits out the whole of the leader's model load before HELLO.
    //                   A leader that DIES is still detected immediately - the socket closes and
    //                   recv returns 0 - so blocking is safe here and a timeout is not.
    int                             timeout_ms_     = 0;
    int                             msg_timeout_ms_ = 0;
    pipe_tp_wire_dtype              dtype_      = PIPE_TP_DTYPE_F32;
    pipe_tp_stats                   stats_;
    bool                            print_stats_at_exit_ = false;
};
