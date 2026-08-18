#pragma once

#include "pipe-dense-segment-manifest.h"
#include "pipe-channel.h"
#include "pipe-protocol.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pipe_dense_segment_client {

std::string prompt_identity(const int32_t * tokens, size_t n_tokens);

class client {
public:
    // need_nextn: whether this head will actually READ the terminal segment's nextn
    // sideband. True only under --spec-type draft-mtp. When false the tail is told at
    // HELLO not to serialize it and the arm pays ZERO nextn bytes per token; when true
    // the tail may still dedup it against the terminal hidden state (see
    // pipe_segment_fwd_resp::nextn_aliased) and this client rebuilds it transparently,
    // so finish_forward()'s response always carries a full nextn run when one was asked
    // for.
    //
    // Deliberately not defaulted: a default would let a new call site silently declare
    // "no need" and starve a draft-mtp run of the sideband it verifies against -- which
    // degrades the acceptance rate without changing a single verified token, so no
    // parity test could catch it. Same reasoning as pipe_decode_segment_fwd_resp's
    // undefaulted tap arguments.
    client(pipe_dense_segment::manifest manifest, uint32_t n_vocab, bool need_nextn);
    ~client();

    client(const client &) = delete;
    client & operator=(const client &) = delete;

    const pipe_dense_segment::manifest & manifest() const;
    bool has_remote_segments() const;

    // PIPE_SEGMENT_TERMINAL_LOGITS or PIPE_SEGMENT_TERMINAL_HIDDEN, negotiated at
    // HELLO. Defaults to HIDDEN (logits-on-head); set WP_SEGMENT_TAIL_LOGITS=1 in
    // the head's environment to request the legacy logits-on-tail path for A/B.
    // NOTE: the tail worker's own WP_SEGMENT_TAIL_LOGITS must agree -- it decides
    // at load time whether to build the LM head into its graph, so a mismatch is
    // rejected at HELLO rather than papered over.
    uint32_t terminal_kind() const;

    // f32 columns per token in the terminal response: n_vocab under LOGITS,
    // n_embd under HIDDEN.
    uint32_t terminal_width() const;

    // f32 columns per token the tail will return in the nextn sideband: n_embd when
    // need_nextn was requested, 0 otherwise. 0 means zero nextn bytes on the wire.
    uint32_t nextn_width() const;

    // One interior tap harvested during the last finish_forward(): the target layer id,
    // the row width, and [n_tokens, width] f32 rows in batch order.
    struct segment_tap {
        uint32_t           layer = 0;
        uint32_t           width = 0;
        std::vector<float> rows;
    };

    // Taps collected across ALL hops of the last forward, ascending by segment then by
    // layer. Valid until the next begin_forward(). Empty when the manifest declares no
    // taps, which is the case for every configuration that does not run a DFlash/DSpark
    // draft against a split target.
    const std::vector<segment_tap> & taps() const;
    uint32_t recurrent_snapshots() const;

    void reset(uint64_t session_id, uint64_t cache_epoch);
    void trim(uint64_t session_id, uint64_t cache_epoch, uint32_t n_past);

    // The prefix is usable only if every remote cache reports exactly n_past.
    // A miss is returned to the caller so it can reset the local head and all
    // remote stages as one conservative operation.
    bool prompt_reuse(uint64_t session_id, uint64_t cache_epoch, uint32_t n_past,
                      const std::string & prompt_identity_sha256);

    // Start a forward at the local-head boundary. finish_forward() receives
    // the first response and drives the remaining remote stages in order.
    void begin_forward(uint64_t session_id, uint64_t seq_id,
                       uint32_t n_tokens, const std::vector<int32_t> & positions,
                       const std::vector<float> & activations);

    // The terminal response has terminal_width() columns; a non-terminal
    // response has manifest().n_embd columns.
    pipe_segment_fwd_resp finish_forward();

private:
    pipe_dense_segment::manifest manifest_value;
    uint32_t requested_terminal_kind = PIPE_SEGMENT_TERMINAL_HIDDEN;
    uint32_t terminal_n_width = 0;
    bool     want_nextn = false;
    uint32_t nextn_n_width = 0;
    uint32_t min_rs_snapshots = 0;
    std::vector<segment_tap> collected_taps;
    std::vector<std::unique_ptr<pipe_channel::channel>> channels;

    struct pending_forward {
        uint64_t session_id = 0;
        uint64_t seq_id = 0;
        uint32_t n_tokens = 0;
        std::vector<int32_t> positions;
    } pending;

    pipe_channel::channel & channel_at(size_t index);
    pipe_channel::received_frame receive(pipe_channel::channel & channel, uint64_t seq_id,
                                         pipe_frame_type expected);
    std::vector<pipe_segment_ctrl_ack> control_all(const pipe_segment_ctrl & control);
};

} // namespace pipe_dense_segment_client
