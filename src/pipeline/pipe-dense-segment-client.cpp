#include "pipe-dense-segment-client.h"

#include "pipe-channel.h"
extern "C" {
#include "sha256/sha256.h"
}

#include <array>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace pipe_dense_segment_client {
namespace {

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("dense segment client: " + message);
}

void throw_peer_error(const pipe_channel::received_frame & frame) {
    if (frame.type != PIPE_ERROR) {
        return;
    }
    const pipe_error error = pipe_decode_error(frame.payload.data(), frame.payload.size());
    fail("peer rejected request: " + error.msg);
}

} // namespace

std::string prompt_identity(const int32_t * tokens, size_t n_tokens) {
    if (tokens == nullptr && n_tokens != 0) {
        fail("cannot hash a null prompt prefix");
    }
    sha256_t hash;
    sha256_init(&hash);
    for (size_t i = 0; i < n_tokens; ++i) {
        const uint32_t value = (uint32_t) tokens[i];
        const unsigned char bytes[4] = {
            (unsigned char) (value >> 0),
            (unsigned char) (value >> 8),
            (unsigned char) (value >> 16),
            (unsigned char) (value >> 24),
        };
        sha256_update(&hash, bytes, sizeof(bytes));
    }
    std::array<unsigned char, SHA256_DIGEST_SIZE> digest{};
    sha256_final(&hash, digest.data());
    std::ostringstream result;
    result << "sha256:" << std::hex << std::setfill('0');
    for (unsigned char byte : digest) {
        result << std::setw(2) << (unsigned int) byte;
    }
    return result.str();
}

client::client(pipe_dense_segment::manifest manifest, uint32_t n_vocab, bool need_nextn) :
        manifest_value(std::move(manifest)), want_nextn(need_nextn) {
    if (manifest_value.segments.empty() || manifest_value.segments.front().layer_first != 0 || n_vocab == 0) {
        fail("invalid local head configuration");
    }

    // Logits-on-head is the default. WP_SEGMENT_TAIL_LOGITS=1 restores the legacy
    // path where the tail projects and ships n_vocab floats per token, which is the
    // A/B control arm -- it must be set identically on the tail worker.
    {
        const char * force_logits = std::getenv("WP_SEGMENT_TAIL_LOGITS");
        requested_terminal_kind = (force_logits != nullptr && force_logits[0] == '1')
            ? (uint32_t) PIPE_SEGMENT_TERMINAL_LOGITS
            : (uint32_t) PIPE_SEGMENT_TERMINAL_HIDDEN;
    }

    for (size_t i = 1; i < manifest_value.segments.size(); ++i) {
        const auto & segment = manifest_value.segments[i];
        channels.emplace_back(std::make_unique<pipe_channel::channel>(pipe_channel::endpoint{
            segment.target.host, segment.target.port,
        }));
    }

    for (size_t i = 0; i < channels.size(); ++i) {
        const auto & segment = manifest_value.segments[i + 1];
        pipe_segment_hello hello;
        hello.segment_id = segment.id;
        hello.layer_first = segment.layer_first;
        hello.layer_last = segment.layer_last;
        hello.model_identity_sha256 = manifest_value.model_identity_sha256;
        hello.n_embd = manifest_value.n_embd;
        hello.n_vocab = n_vocab;
        hello.wire_precision = PIPE_SEGMENT_WIRE_F32;
        hello.capabilities = PIPE_SEGMENT_CAP_FWD | PIPE_SEGMENT_CAP_RESET |
            PIPE_SEGMENT_CAP_KV_TRIM | PIPE_SEGMENT_CAP_PROMPT_REUSE;
        hello.terminal_kind = requested_terminal_kind;
        // Interior taps come from the manifest, which the worker parses too, so the
        // worker can arm extraction as a LOAD-TIME graph decision and HELLO only has to
        // confirm the two sides agree -- the same division of labour as terminal_kind.
        hello.tap_layers = segment.tap_layers;
        // NEXTN SIDEBAND. Only the TERMINAL segment produces one, and only this head
        // knows whether anything will read it, so it is declared here per hop rather
        // than derived from the manifest -- the manifest describes the topology, and the
        // need is a property of the speculative arm the server was started with.
        hello.nextn_need = (want_nextn && i + 1 == manifest_value.segments.size() - 1) ? 1u : 0u;
        channel_at(i).send_frame(PIPE_SEGMENT_HELLO, 0, pipe_encode_segment_hello(hello));
    }
    for (auto & channel : channels) {
        channel->flush();
    }
    for (size_t i = 0; i < channels.size(); ++i) {
        const pipe_channel::received_frame frame = receive(channel_at(i), 0, PIPE_SEGMENT_HELLO_ACK);
        const pipe_segment_hello_ack ack =
            pipe_decode_segment_hello_ack(frame.payload.data(), frame.payload.size());
        if (!ack.accepted) {
            fail("segment " + std::to_string(manifest_value.segments[i + 1].id) +
                 " rejected HELLO: " + ack.reason);
        }
        if (ack.n_vocab != n_vocab) {
            fail("segment " + std::to_string(manifest_value.segments[i + 1].id) +
                 " returned a different n_vocab");
        }
        if (ack.terminal_kind != requested_terminal_kind) {
            fail("segment " + std::to_string(manifest_value.segments[i + 1].id) +
                 " negotiated terminal kind " + std::to_string(ack.terminal_kind) +
                 " but the head requested " + std::to_string(requested_terminal_kind) +
                 " (WP_SEGMENT_TAIL_LOGITS must match on both sides)");
        }
        // Interior taps: the echo must match exactly, and a non-empty list must come
        // with the width the head will slice it by. Anything softer and a segment that
        // quietly serves no taps leaves the head conditioning its draft on a stale
        // buffer -- which changes no verified token and so survives a parity test.
        {
            const auto & want = manifest_value.segments[i + 1].tap_layers;
            if (ack.tap_layers != want) {
                fail("segment " + std::to_string(manifest_value.segments[i + 1].id) +
                     " negotiated " + std::to_string(ack.tap_layers.size()) +
                     " interior taps but the manifest declares " + std::to_string(want.size()) +
                     " (the head and the worker must be given the same manifest)");
            }
            if (!want.empty() && ack.tap_width != (uint32_t) manifest_value.n_embd) {
                fail("segment " + std::to_string(manifest_value.segments[i + 1].id) +
                     " reported interior tap width " + std::to_string(ack.tap_width) +
                     " but the manifest n_embd is " + std::to_string(manifest_value.n_embd));
            }
        }

        // NEXTN SIDEBAND: the segment must answer with exactly the width our declared
        // need implies. Anything else and the forward response would carry a run the
        // decoder's length check does not expect, so pin it at the handshake where the
        // message can name both sides.
        {
            const bool is_tail = i + 1 == manifest_value.segments.size() - 1;
            const uint32_t expected = (want_nextn && is_tail) ? (uint32_t) manifest_value.n_embd : 0u;
            if (ack.nextn_width != expected) {
                fail("segment " + std::to_string(manifest_value.segments[i + 1].id) +
                     " reported nextn width " + std::to_string(ack.nextn_width) +
                     " but the head declared need=" + std::to_string(expected != 0 ? 1 : 0) +
                     ", which implies " + std::to_string(expected));
            }
        }

        if (i == 0 || ack.rs_snapshots < min_rs_snapshots) {
            min_rs_snapshots = ack.rs_snapshots;
        }
        if (i + 1 == manifest_value.segments.size() - 1) {
            // The tail's declared width is authoritative; cross-check it against
            // what the negotiated kind implies so a width/kind disagreement cannot
            // reach the decode path as a silent mis-slice.
            const uint32_t expected = requested_terminal_kind == PIPE_SEGMENT_TERMINAL_LOGITS
                ? n_vocab : (uint32_t) manifest_value.n_embd;
            if (ack.terminal_width != expected) {
                fail("segment " + std::to_string(manifest_value.segments[i + 1].id) +
                     " reported terminal width " + std::to_string(ack.terminal_width) +
                     " but the negotiated kind implies " + std::to_string(expected));
            }
            terminal_n_width = ack.terminal_width;
            nextn_n_width    = ack.nextn_width;
        }
    }
}

client::~client() = default;

const pipe_dense_segment::manifest & client::manifest() const {
    return manifest_value;
}

bool client::has_remote_segments() const {
    return !channels.empty();
}

uint32_t client::terminal_kind() const {
    return requested_terminal_kind;
}

uint32_t client::terminal_width() const {
    return terminal_n_width;
}

uint32_t client::nextn_width() const {
    return nextn_n_width;
}

const std::vector<client::segment_tap> & client::taps() const {
    return collected_taps;
}

uint32_t client::recurrent_snapshots() const {
    return min_rs_snapshots;
}

pipe_channel::channel & client::channel_at(size_t index) {
    return *channels.at(index);
}

pipe_channel::received_frame client::receive(
        pipe_channel::channel & channel, uint64_t seq_id, pipe_frame_type expected) {
    pipe_channel::received_frame frame;
    if (!pipe_channel::channel::harvest({ &channel }, frame, -1)) {
        fail("segment channel returned no frame");
    }
    throw_peer_error(frame);
    if (frame.type != expected || frame.seq_id != seq_id) {
        fail("segment returned an unexpected frame");
    }
    return frame;
}

std::vector<pipe_segment_ctrl_ack> client::control_all(const pipe_segment_ctrl & control) {
    std::vector<uint64_t> sequence_ids;
    sequence_ids.reserve(channels.size());
    for (auto & channel : channels) {
        sequence_ids.push_back(channel->send_request(PIPE_SEGMENT_CTRL, pipe_encode_segment_ctrl(control)));
    }
    for (auto & channel : channels) {
        channel->flush();
    }

    std::vector<pipe_segment_ctrl_ack> result;
    result.reserve(channels.size());
    for (size_t i = 0; i < channels.size(); ++i) {
        const pipe_channel::received_frame frame =
            receive(channel_at(i), sequence_ids[i], PIPE_SEGMENT_CTRL_ACK);
        result.push_back(pipe_decode_segment_ctrl_ack(frame.payload.data(), frame.payload.size()));
    }
    return result;
}

void client::reset(uint64_t session_id, uint64_t cache_epoch) {
    const pipe_segment_ctrl control = {
        PIPE_SEGMENT_CTRL_VERSION, PIPE_SEGMENT_CTRL_RESET, session_id, cache_epoch, 0, "",
    };
    for (const pipe_segment_ctrl_ack & ack : control_all(control)) {
        if (ack.control != control.control || ack.session_id != session_id || ack.cache_epoch != cache_epoch ||
            ack.status != PIPE_SEGMENT_CTRL_APPLIED || ack.n_past != 0) {
            fail("segment did not acknowledge RESET");
        }
    }
}

void client::trim(uint64_t session_id, uint64_t cache_epoch, uint32_t n_past) {
    const pipe_segment_ctrl control = {
        PIPE_SEGMENT_CTRL_VERSION, PIPE_SEGMENT_CTRL_KV_TRIM, session_id, cache_epoch, n_past, "",
    };
    for (const pipe_segment_ctrl_ack & ack : control_all(control)) {
        if (ack.control != control.control || ack.session_id != session_id || ack.cache_epoch != cache_epoch ||
            ack.status != PIPE_SEGMENT_CTRL_APPLIED || ack.n_past != n_past) {
            fail("segment did not acknowledge KV_TRIM");
        }
    }
}

bool client::prompt_reuse(uint64_t session_id, uint64_t cache_epoch, uint32_t n_past,
                          const std::string & prompt_identity_sha256) {
    if (prompt_identity_sha256.empty()) {
        fail("cannot query prompt reuse without a prompt identity");
    }
    const pipe_segment_ctrl control = {
        PIPE_SEGMENT_CTRL_VERSION, PIPE_SEGMENT_CTRL_PROMPT_REUSE,
        session_id, cache_epoch, n_past, prompt_identity_sha256,
    };
    for (const pipe_segment_ctrl_ack & ack : control_all(control)) {
        if (ack.control != control.control || ack.session_id != session_id || ack.cache_epoch != cache_epoch ||
            ack.status != PIPE_SEGMENT_CTRL_APPLIED || ack.n_past != n_past) {
            return false;
        }
    }
    return true;
}

void client::begin_forward(
        uint64_t session_id, uint64_t seq_id, uint32_t n_tokens,
        const std::vector<int32_t> & positions, const std::vector<float> & activations) {
    if (channels.empty()) {
        fail("begin_forward requires at least one remote segment");
    }
    if (pending.seq_id != 0 || seq_id == 0 || n_tokens == 0 || positions.size() != n_tokens ||
        activations.size() != (size_t) n_tokens * manifest_value.n_embd) {
        fail("invalid begin_forward boundary");
    }

    pipe_segment_fwd_req request;
    request.session_id = session_id;
    request.seq_id = seq_id;
    request.n_tokens = n_tokens;
    request.positions = positions;
    request.seq_token_counts = { n_tokens };
    request.activations = activations;
    channel_at(0).send_frame(PIPE_SEGMENT_FWD_REQ, seq_id,
        pipe_encode_segment_fwd_req(request, manifest_value.n_embd));
    channel_at(0).flush();

    pending = { session_id, seq_id, n_tokens, positions };
}

pipe_segment_fwd_resp client::finish_forward() {
    if (pending.seq_id == 0) {
        fail("finish_forward without begin_forward");
    }

    try {
        // Interior taps must be harvested per hop. `response` below is overwritten on
        // every iteration and only the terminal one survives to the caller, so a tap
        // returned by a MIDDLE segment -- which is the whole point of the feature --
        // would otherwise be discarded before finish_forward() returns.
        collected_taps.clear();

        pipe_segment_fwd_resp response;
        for (size_t i = 0; i < channels.size(); ++i) {
            const bool is_terminal = i + 1 == channels.size();
            const int32_t output_width = is_terminal ? (int32_t) terminal_n_width : manifest_value.n_embd;
            // Was hardcoded to n_embd on the terminal hop; it is now whatever the HELLO
            // negotiated, which is 0 on every arm that does not read the sideband.
            const int32_t nextn_width = is_terminal ? (int32_t) nextn_n_width : 0;
            const auto & seg_taps = manifest_value.segments[i + 1].tap_layers;
            const int32_t tap_width = seg_taps.empty() ? 0 : manifest_value.n_embd;
            const pipe_channel::received_frame frame =
                receive(channel_at(i), pending.seq_id, PIPE_SEGMENT_FWD_RESP);
            response = pipe_decode_segment_fwd_resp(
                frame.payload.data(), frame.payload.size(), output_width, nextn_width,
                tap_width, (int32_t) seg_taps.size());
            if (response.session_id != pending.session_id || response.seq_id != pending.seq_id ||
                response.n_tokens != pending.n_tokens) {
                fail("segment returned a stale forward response");
            }
            // NEXTN DEDUP: the tail found its nextn sideband bit-identical to the
            // terminal hidden state and shipped one copy. Rebuild the second here so
            // every caller above this line sees the same response it always did -- the
            // saving is on the wire, not in the API. The decoder has already checked
            // nextn_width == output_width, so this is an exact copy, never a reshape.
            if (response.nextn_aliased != 0) {
                response.nextn = response.activations;
            }
            // Split this hop's concatenated tap blocks back out per layer, in the
            // ascending order the HELLO negotiated.
            for (size_t k = 0; k < seg_taps.size(); ++k) {
                const size_t rows = (size_t) response.n_tokens * (size_t) tap_width;
                const float * src = response.taps.data() + k * rows;
                collected_taps.push_back(segment_tap{
                    seg_taps[k], (uint32_t) tap_width,
                    std::vector<float>(src, src + rows),
                });
            }
            if (!is_terminal) {
                pipe_segment_fwd_req request;
                request.session_id = pending.session_id;
                request.seq_id = pending.seq_id;
                request.n_tokens = pending.n_tokens;
                request.positions = pending.positions;
                request.seq_token_counts = { pending.n_tokens };
                request.activations = std::move(response.activations);
                channel_at(i + 1).send_frame(PIPE_SEGMENT_FWD_REQ, pending.seq_id,
                    pipe_encode_segment_fwd_req(request, manifest_value.n_embd));
                channel_at(i + 1).flush();
            }
        }
        pending = {};
        return response;
    } catch (...) {
        pending = {};
        throw;
    }
}

} // namespace pipe_dense_segment_client
