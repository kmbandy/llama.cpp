#pragma once

#include "pipe-protocol.h"
#include "pipe-transport.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pipe_channel {

struct endpoint {
    std::string host;
    int         port = 0;
};

struct frame {
    pipe_frame_type      type = PIPE_ERROR;
    uint64_t             seq_id = 0;
    std::vector<uint8_t> payload;
};

struct received_frame : frame {
    class channel * source = nullptr;
};

// One persistent, full-duplex pipeline connection. Requests are assigned a
// monotonically increasing non-zero sequence id. Frames with explicit ids are
// for handshake/control paths, including the reserved HELLO id 0.
class channel {
  public:
    explicit channel(endpoint target);
    explicit channel(pipe_socket_ptr socket, std::string peer_name = {});
    ~channel();

    channel(const channel &) = delete;
    channel & operator=(const channel &) = delete;
    channel(channel &&) noexcept;
    channel & operator=(channel &&) noexcept;

    uint64_t send_request(pipe_frame_type type, std::vector<uint8_t> payload);
    void send_frame(pipe_frame_type type, uint64_t seq_id, std::vector<uint8_t> payload);

    // Wait until the per-socket FIFO has reached the wire. This does not await
    // a peer response.
    void flush();

    int poll_fd() const;
    const std::string & peer_name() const;

    // Harvest one frame from any readable channel. A timeout of -1 waits
    // forever, zero polls, and a positive value is milliseconds. At most one
    // thread may harvest a particular channel at a time.
    static bool harvest(const std::vector<channel *> & channels, received_frame & out,
                        int timeout_ms);

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

} // namespace pipe_channel
