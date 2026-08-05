#include "pipe-protocol.h"
#include "pipe-transport.h"

#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

// ---------------------------------------------------------------------------
// little-endian read/write helpers (byte-explicit; no struct memcpy, no
// host-endianness assumptions)

static void wr_u32(uint8_t * & p, uint32_t v) {
    p[0] = (uint8_t) (v >>  0);
    p[1] = (uint8_t) (v >>  8);
    p[2] = (uint8_t) (v >> 16);
    p[3] = (uint8_t) (v >> 24);
    p += 4;
}

static void wr_u64(uint8_t * & p, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        p[i] = (uint8_t) (v >> (8 * i));
    }
    p += 8;
}

static void wr_u16(uint8_t * & p, uint16_t v) {
    p[0] = (uint8_t) (v >> 0);
    p[1] = (uint8_t) (v >> 8);
    p += 2;
}

static void wr_i32(uint8_t * & p, int32_t v) {
    wr_u32(p, (uint32_t) v);
}

static void wr_f32(uint8_t * & p, float v) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(v), "f32 wire size");
    std::memcpy(&bits, &v, sizeof(bits));
    wr_u32(p, bits);
}

static uint32_t rd_u32(const uint8_t * & p) {
    uint32_t v = (uint32_t) p[0]
               | ((uint32_t) p[1] <<  8)
               | ((uint32_t) p[2] << 16)
               | ((uint32_t) p[3] << 24);
    p += 4;
    return v;
}

static uint64_t rd_u64(const uint8_t * & p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= ((uint64_t) p[i]) << (8 * i);
    }
    p += 8;
    return v;
}

static uint16_t rd_u16(const uint8_t * & p) {
    const uint16_t v = (uint16_t) p[0] | ((uint16_t) p[1] << 8);
    p += 2;
    return v;
}

static int32_t rd_i32(const uint8_t * & p) {
    return (int32_t) rd_u32(p);
}

static float rd_f32(const uint8_t * & p) {
    const uint32_t bits = rd_u32(p);
    float          value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// ---------------------------------------------------------------------------
// error helper

[[noreturn]] static void fail(pipe_error_code code, const char * fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    throw pipe_protocol_error(code, buf);
}

uint32_t pipe_hidden_elt_size(int32_t hidden_type) {
    switch (hidden_type) {
        case PIPE_HIDDEN_F32: return 4;
        case PIPE_HIDDEN_F16: return 2;
        default:
            fail(PIPE_ERR_HELLO, "pipe: unknown hidden_type %d", (int) hidden_type);
    }
}

// ---------------------------------------------------------------------------
// header

void pipe_encode_header(uint8_t out[PIPE_HEADER_SIZE], const pipe_frame_header & h) {
    uint8_t * p = out;
    wr_u32(p, h.magic);
    wr_u32(p, h.version);
    wr_u32(p, h.type);
    wr_u32(p, h.flags);
    wr_u64(p, h.seq_id);
    wr_u64(p, h.length);
}

pipe_frame_header pipe_decode_header(const uint8_t in[PIPE_HEADER_SIZE]) {
    const uint8_t * p = in;
    pipe_frame_header h;
    h.magic   = rd_u32(p);
    h.version = rd_u32(p);
    h.type    = rd_u32(p);
    h.flags   = rd_u32(p);
    h.seq_id  = rd_u64(p);
    h.length  = rd_u64(p);

    if (h.magic != PIPE_MAGIC) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: bad frame magic 0x%08x (want 0x%08x)",
             h.magic, PIPE_MAGIC);
    }
    if (h.version != PIPE_VERSION) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: unsupported protocol version %u (want %u)",
             h.version, PIPE_VERSION);
    }
    if (h.length > PIPE_MAX_PAYLOAD) {
        fail(PIPE_ERR_BAD_FRAME,
             "pipe: frame length %llu exceeds max payload %llu; refusing to allocate",
             (unsigned long long) h.length, (unsigned long long) PIPE_MAX_PAYLOAD);
    }
    return h;
}

// ---------------------------------------------------------------------------
// HELLO

std::vector<uint8_t> pipe_encode_hello(const pipe_hello & p) {
    std::vector<uint8_t> out(pipe_hello::WIRE_SIZE);
    uint8_t * w = out.data();
    wr_u32(w, p.role);
    wr_i32(w, p.layer_first);
    wr_i32(w, p.layer_last);
    wr_i32(w, p.n_layer);
    wr_i32(w, p.n_embd);
    wr_i32(w, p.hidden_type);
    wr_u32(w, p.model_hash);
    return out;
}

pipe_hello pipe_decode_hello(const uint8_t * buf, size_t len) {
    if (len != pipe_hello::WIRE_SIZE) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: HELLO payload %zu bytes, want %zu",
             len, pipe_hello::WIRE_SIZE);
    }
    const uint8_t * p = buf;
    pipe_hello h;
    h.role        = rd_u32(p);
    h.layer_first = rd_i32(p);
    h.layer_last  = rd_i32(p);
    h.n_layer     = rd_i32(p);
    h.n_embd      = rd_i32(p);
    h.hidden_type = rd_i32(p);
    h.model_hash  = rd_u32(p);

    if (h.role > PIPE_ROLE_TAIL) {
        fail(PIPE_ERR_HELLO, "pipe: HELLO role %u out of range", h.role);
    }
    // hidden_type validity is enforced via elt_size in validate_hello; an
    // unknown type there throws before any tensor buffer is sized from it.
    return h;
}

// ---------------------------------------------------------------------------
// expert HELLO
//
//   u32 tag = "EXP2"
//   u32 role
//   i32 hidden_type
//   i32 n_embd, n_ff_exp, n_expert, n_expert_used
//   i32 expert_first, expert_last
//   u32 n_slots
//   u32 n_layers
//   i32 layers[n_layers]
//   u32 model_identity_len
//   u8  model_identity[model_identity_len]
//   u32 shard_identity_len
//   u8  shard_identity[shard_identity_len]

static constexpr uint32_t PIPE_EXPERT_HELLO_TAG = 0x32505845u; // "EXP2"

std::vector<uint8_t> pipe_encode_expert_hello(const pipe_expert_hello & p) {
    if (p.model_identity.size() > std::numeric_limits<uint32_t>::max() ||
        p.shard_identity.size() > std::numeric_limits<uint32_t>::max()) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO identity is too long");
    }
    const uint64_t total = 13 * 4ull + (uint64_t) p.layers.size() * 4ull +
                           p.model_identity.size() + p.shard_identity.size();
    if (total > PIPE_MAX_PAYLOAD) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO encode size %llu exceeds max payload",
             (unsigned long long) total);
    }

    std::vector<uint8_t> out((size_t) total);
    uint8_t * w = out.data();
    wr_u32(w, PIPE_EXPERT_HELLO_TAG);
    wr_u32(w, p.role);
    wr_i32(w, p.hidden_type);
    wr_i32(w, p.n_embd);
    wr_i32(w, p.n_ff_exp);
    wr_i32(w, p.n_expert);
    wr_i32(w, p.n_expert_used);
    wr_i32(w, p.expert_first);
    wr_i32(w, p.expert_last);
    wr_u32(w, p.n_slots);
    wr_u32(w, (uint32_t) p.layers.size());
    for (int32_t layer : p.layers) {
        wr_i32(w, layer);
    }
    wr_u32(w, (uint32_t) p.model_identity.size());
    if (!p.model_identity.empty()) {
        std::memcpy(w, p.model_identity.data(), p.model_identity.size());
        w += p.model_identity.size();
    }
    wr_u32(w, (uint32_t) p.shard_identity.size());
    if (!p.shard_identity.empty()) {
        std::memcpy(w, p.shard_identity.data(), p.shard_identity.size());
    }
    return out;
}

pipe_expert_hello pipe_decode_expert_hello(const uint8_t * buf, size_t len) {
    if (len < 13 * 4ull) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO payload %zu bytes too small", len);
    }
    const uint8_t * p   = buf;
    const uint8_t * end = buf + len;
    if (rd_u32(p) != PIPE_EXPERT_HELLO_TAG) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO has the wrong payload tag");
    }

    pipe_expert_hello h;
    h.role          = rd_u32(p);
    h.hidden_type   = rd_i32(p);
    h.n_embd        = rd_i32(p);
    h.n_ff_exp      = rd_i32(p);
    h.n_expert      = rd_i32(p);
    h.n_expert_used = rd_i32(p);
    h.expert_first  = rd_i32(p);
    h.expert_last   = rd_i32(p);
    h.n_slots       = rd_u32(p);
    const uint32_t n_layers = rd_u32(p);

    if (h.role > PIPE_EXPERT_ROLE_WORKER) {
        fail(PIPE_ERR_HELLO, "pipe: expert HELLO role %u out of range", h.role);
    }
    if (h.hidden_type != PIPE_HIDDEN_F16) {
        fail(PIPE_ERR_HELLO, "pipe: expert HELLO hidden type %d is not F16", h.hidden_type);
    }
    if (h.n_embd <= 0 || h.n_ff_exp <= 0 || h.n_expert <= 0 || h.n_expert_used <= 0 ||
        h.n_expert_used > h.n_expert) {
        fail(PIPE_ERR_HELLO, "pipe: expert HELLO has invalid hparams");
    }
    if ((uint64_t) (end - p) < (uint64_t) n_layers * 4ull + 8ull) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO layer array is truncated");
    }
    h.layers.reserve(n_layers);
    std::set<int32_t> seen_layers;
    for (uint32_t i = 0; i < n_layers; ++i) {
        const int32_t layer = rd_i32(p);
        if (layer < 0 || !seen_layers.insert(layer).second) {
            fail(PIPE_ERR_HELLO, "pipe: expert HELLO has an invalid or repeated layer");
        }
        h.layers.push_back(layer);
    }

    const uint32_t identity_len = rd_u32(p);
    if ((uint64_t) (end - p) < (uint64_t) identity_len + 4ull) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO model identity is truncated");
    }
    h.model_identity.assign((const char *) p, identity_len);
    p += identity_len;
    const uint32_t shard_identity_len = rd_u32(p);
    if ((uint64_t) (end - p) != shard_identity_len) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO shard identity bytes %lld, want %u",
             (long long) (end - p), shard_identity_len);
    }
    h.shard_identity.assign((const char *) p, shard_identity_len);
    if (h.model_identity.empty()) {
        fail(PIPE_ERR_HELLO, "pipe: expert HELLO model identity is empty");
    }
    if (h.shard_identity.empty()) {
        fail(PIPE_ERR_HELLO, "pipe: expert HELLO shard identity is empty");
    }

    if (h.role == PIPE_EXPERT_ROLE_WORKER) {
        if (h.layers.empty() || h.expert_first < 0 || h.expert_last < h.expert_first ||
            h.expert_last >= h.n_expert || h.n_slots == 0) {
            fail(PIPE_ERR_HELLO, "pipe: expert worker HELLO has an invalid service range");
        }
    }
    return h;
}

// ---------------------------------------------------------------------------
// expert HELLO acknowledgement
//
//   u32 tag = "EXA1"
//   u32 accepted
//   u32 reason_len
//   u8  reason[reason_len]

static constexpr uint32_t PIPE_EXPERT_HELLO_ACK_TAG = 0x31415845u; // "EXA1"

std::vector<uint8_t> pipe_encode_expert_hello_ack(const pipe_expert_hello_ack & p) {
    if (p.reason.size() > std::numeric_limits<uint32_t>::max()) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO acknowledgement reason is too long");
    }
    if (p.accepted && !p.reason.empty()) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: accepted expert HELLO has a rejection reason");
    }
    if (!p.accepted && p.reason.empty()) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: rejected expert HELLO has no reason");
    }
    const uint64_t total = 12ull + p.reason.size();
    if (total > PIPE_MAX_PAYLOAD) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO acknowledgement exceeds max payload");
    }

    std::vector<uint8_t> out((size_t) total);
    uint8_t * w = out.data();
    wr_u32(w, PIPE_EXPERT_HELLO_ACK_TAG);
    wr_u32(w, p.accepted ? 1u : 0u);
    wr_u32(w, (uint32_t) p.reason.size());
    if (!p.reason.empty()) {
        std::memcpy(w, p.reason.data(), p.reason.size());
    }
    return out;
}

pipe_expert_hello_ack pipe_decode_expert_hello_ack(const uint8_t * buf, size_t len) {
    if (len < 12) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO acknowledgement payload is too small");
    }
    const uint8_t * p   = buf;
    const uint8_t * end = buf + len;
    if (rd_u32(p) != PIPE_EXPERT_HELLO_ACK_TAG) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO acknowledgement has the wrong payload tag");
    }

    const uint32_t accepted = rd_u32(p);
    if (accepted > 1) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO acknowledgement has an invalid result");
    }
    const uint32_t reason_len = rd_u32(p);
    if ((uint64_t) (end - p) != reason_len) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert HELLO acknowledgement reason bytes %lld, want %u",
             (long long) (end - p), reason_len);
    }

    pipe_expert_hello_ack result;
    result.accepted = accepted != 0;
    result.reason.assign((const char *) p, reason_len);
    if (result.accepted && !result.reason.empty()) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: accepted expert HELLO has a rejection reason");
    }
    if (!result.accepted && result.reason.empty()) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: rejected expert HELLO has no reason");
    }
    return result;
}

// ---------------------------------------------------------------------------
// expert dispatch request

std::vector<uint8_t> pipe_encode_expert_dispatch_req(const pipe_expert_dispatch_req & p) {
    if (p.n_tokens == 0 || p.assignments.empty()) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert dispatch requires tokens and assignments");
    }
    if (!std::isfinite(p.swiglu_clamp) || p.swiglu_clamp < 0.0f) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert dispatch has an invalid swiglu clamp");
    }
    // 16, not 12: + f32 swiglu_clamp as of PIPE_VERSION 3.
    uint64_t total = 16;
    for (const pipe_expert_assignment & assignment : p.assignments) {
        if (assignment.weights.size() != p.n_tokens) {
            fail(PIPE_ERR_BAD_FRAME, "pipe: expert dispatch weight count does not match n_tokens");
        }
        total += 4ull + (uint64_t) assignment.weights.size() * 4ull;
    }
    total += (uint64_t) p.activations.size() * 2ull;
    if (total > PIPE_MAX_PAYLOAD) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert dispatch encode size %llu exceeds max payload",
             (unsigned long long) total);
    }

    std::vector<uint8_t> out((size_t) total);
    uint8_t * w = out.data();
    wr_i32(w, p.layer);
    wr_u32(w, p.n_tokens);
    wr_u32(w, (uint32_t) p.assignments.size());
    wr_f32(w, p.swiglu_clamp);
    for (const pipe_expert_assignment & assignment : p.assignments) {
        wr_i32(w, assignment.expert_id);
        for (float weight : assignment.weights) {
            wr_f32(w, weight);
        }
    }
    for (uint16_t value : p.activations) {
        wr_u16(w, value);
    }
    return out;
}

pipe_expert_dispatch_req pipe_decode_expert_dispatch_req(
        const uint8_t * buf, size_t len, int32_t n_embd) {
    if (n_embd <= 0 || len < 16) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert dispatch payload is too small");
    }
    const uint8_t * p   = buf;
    const uint8_t * end = buf + len;

    pipe_expert_dispatch_req r;
    r.layer               = rd_i32(p);
    r.n_tokens            = rd_u32(p);
    const uint32_t n_assignments = rd_u32(p);
    r.swiglu_clamp        = rd_f32(p);
    if (r.layer < 0 || r.n_tokens == 0 || n_assignments == 0) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert dispatch has invalid dimensions");
    }
    if (!std::isfinite(r.swiglu_clamp) || r.swiglu_clamp < 0.0f) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert dispatch has an invalid swiglu clamp");
    }

    const uint64_t assignment_bytes =
        (uint64_t) n_assignments * (4ull + (uint64_t) r.n_tokens * 4ull);
    const uint64_t activation_bytes =
        (uint64_t) r.n_tokens * (uint64_t) n_embd * 2ull;
    if ((uint64_t) (end - p) != assignment_bytes + activation_bytes) {
        fail(PIPE_ERR_BAD_FRAME,
             "pipe: expert dispatch payload bytes %lld do not match dimensions",
             (long long) (end - p));
    }

    std::set<int32_t> seen_experts;
    r.assignments.reserve(n_assignments);
    for (uint32_t i = 0; i < n_assignments; ++i) {
        pipe_expert_assignment assignment;
        assignment.expert_id = rd_i32(p);
        if (assignment.expert_id < 0 || !seen_experts.insert(assignment.expert_id).second) {
            fail(PIPE_ERR_BAD_FRAME, "pipe: expert dispatch has an invalid or repeated expert");
        }
        assignment.weights.reserve(r.n_tokens);
        for (uint32_t t = 0; t < r.n_tokens; ++t) {
            const float weight = rd_f32(p);
            if (!std::isfinite(weight)) {
                fail(PIPE_ERR_BAD_FRAME, "pipe: expert dispatch has a non-finite weight");
            }
            assignment.weights.push_back(weight);
        }
        r.assignments.push_back(std::move(assignment));
    }

    const size_t n_activations = (size_t) r.n_tokens * (size_t) n_embd;
    r.activations.reserve(n_activations);
    for (size_t i = 0; i < n_activations; ++i) {
        r.activations.push_back(rd_u16(p));
    }
    return r;
}

// ---------------------------------------------------------------------------
// expert partial response

std::vector<uint8_t> pipe_encode_expert_partial(const pipe_expert_partial & p) {
    // 4 bytes per value: partials are f32 as of PIPE_VERSION 2. See the note on
    // pipe_expert_partial -- f16 subtotals made the MoE result depend on the
    // expert->worker partition, which moves with batch width.
    const uint64_t total = 8ull + (uint64_t) p.partial.size() * 4ull;
    if (p.n_tokens == 0 || total > PIPE_MAX_PAYLOAD) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: invalid expert partial response");
    }
    std::vector<uint8_t> out((size_t) total);
    uint8_t * w = out.data();
    wr_i32(w, p.layer);
    wr_u32(w, p.n_tokens);
    for (float value : p.partial) {
        wr_f32(w, value);
    }
    return out;
}

pipe_expert_partial pipe_decode_expert_partial(
        const uint8_t * buf, size_t len, int32_t n_embd) {
    if (n_embd <= 0 || len < 8) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert partial payload is too small");
    }
    const uint8_t * p   = buf;
    const uint8_t * end = buf + len;
    pipe_expert_partial r;
    r.layer    = rd_i32(p);
    r.n_tokens = rd_u32(p);
    const uint64_t n_values = (uint64_t) r.n_tokens * (uint64_t) n_embd;
    if (r.layer < 0 || r.n_tokens == 0 || (uint64_t) (end - p) != n_values * 4ull) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: expert partial dimensions do not match payload");
    }
    r.partial.reserve((size_t) n_values);
    for (uint64_t i = 0; i < n_values; ++i) {
        r.partial.push_back(rd_f32(p));
    }
    return r;
}

// ---------------------------------------------------------------------------
// FWD_REQ
//
//   u32  n_tokens
//   u32  n_pos_per_embd
//   i32  pos[n_tokens * n_pos_per_embd]
//   u32  n_seqs
//   i32  seq_tokens[n_seqs]
//   u8[] hidden, n_embd * n_tokens * elt_size

std::vector<uint8_t> pipe_encode_fwd_req(const pipe_fwd_req & p, int32_t hidden_type) {
    (void) hidden_type; // hidden is already bytes by the time it reaches here
    const uint64_t total = 4 + 4
        + (uint64_t) p.pos.size() * 4
        + 4
        + (uint64_t) p.seq_tokens.size() * 4
        + (uint64_t) p.hidden.size();
    if (total > PIPE_MAX_PAYLOAD) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: FWD_REQ encode size %llu exceeds max payload",
             (unsigned long long) total);
    }
    std::vector<uint8_t> out((size_t) total);
    uint8_t * w = out.data();
    wr_u32(w, p.n_tokens);
    wr_u32(w, p.n_pos_per_embd);
    for (int32_t v : p.pos)        { wr_i32(w, v); }
    wr_u32(w, (uint32_t) p.seq_tokens.size());
    for (int32_t v : p.seq_tokens) { wr_i32(w, v); }
    if (!p.hidden.empty()) {
        std::memcpy(w, p.hidden.data(), p.hidden.size());
    }
    return out;
}

pipe_fwd_req pipe_decode_fwd_req(const uint8_t * buf, size_t len, int32_t n_embd, int32_t hidden_type) {
    const uint32_t elt = pipe_hidden_elt_size(hidden_type);
    if (len < 12) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: FWD_REQ payload %zu bytes too small", len);
    }
    const uint8_t * p   = buf;
    const uint8_t * end = buf + len;

    pipe_fwd_req r;
    r.n_tokens       = rd_u32(p);
    r.n_pos_per_embd = rd_u32(p);
    if (r.n_pos_per_embd == 0) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: FWD_REQ n_pos_per_embd is 0");
    }

    const uint64_t n_pos = (uint64_t) r.n_tokens * r.n_pos_per_embd;
    if ((uint64_t) (end - p) < n_pos * 4) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: FWD_REQ pos array truncated");
    }
    r.pos.reserve((size_t) n_pos);
    for (uint64_t i = 0; i < n_pos; ++i) { r.pos.push_back(rd_i32(p)); }

    if (end - p < 4) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: FWD_REQ missing n_seqs");
    }
    const uint32_t n_seqs = rd_u32(p);
    if ((uint64_t) (end - p) < (uint64_t) n_seqs * 4) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: FWD_REQ seq_tokens array truncated");
    }
    int64_t sum = 0;
    r.seq_tokens.reserve(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        const int32_t v = rd_i32(p);
        if (v < 0) {
            fail(PIPE_ERR_BAD_FRAME, "pipe: FWD_REQ seq_tokens[%u] = %d < 0", i, v);
        }
        sum += v;
        r.seq_tokens.push_back(v);
    }
    if (n_seqs > 0 && sum != (int64_t) r.n_tokens) {
        fail(PIPE_ERR_BAD_FRAME,
             "pipe: FWD_REQ seq_tokens sum %lld != n_tokens %u",
             (long long) sum, r.n_tokens);
    }

    const uint64_t n_hidden = (uint64_t) n_embd * r.n_tokens * elt;
    if ((uint64_t) (end - p) != n_hidden) {
        fail(PIPE_ERR_BAD_FRAME,
             "pipe: FWD_REQ hidden bytes %lld, want %lld (n_embd=%d, n_tokens=%u, elt=%u)",
             (long long) (end - p), (long long) n_hidden, n_embd, r.n_tokens, elt);
    }
    r.hidden.assign(p, end);
    return r;
}

// ---------------------------------------------------------------------------
// FWD_RESP
//
//   u32  n_tokens
//   u8[] hidden, n_embd * n_tokens * elt_size

std::vector<uint8_t> pipe_encode_fwd_resp(const pipe_fwd_resp & p) {
    const uint64_t total = 4 + (uint64_t) p.hidden.size();
    if (total > PIPE_MAX_PAYLOAD) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: FWD_RESP encode size %llu exceeds max payload",
             (unsigned long long) total);
    }
    std::vector<uint8_t> out((size_t) total);
    uint8_t * w = out.data();
    wr_u32(w, p.n_tokens);
    if (!p.hidden.empty()) {
        std::memcpy(w, p.hidden.data(), p.hidden.size());
    }
    return out;
}

pipe_fwd_resp pipe_decode_fwd_resp(const uint8_t * buf, size_t len, int32_t n_embd, int32_t hidden_type) {
    const uint32_t elt = pipe_hidden_elt_size(hidden_type);
    if (len < 4) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: FWD_RESP payload %zu bytes too small", len);
    }
    const uint8_t * p   = buf;
    const uint8_t * end = buf + len;
    pipe_fwd_resp r;
    r.n_tokens = rd_u32(p);
    const uint64_t n_hidden = (uint64_t) n_embd * r.n_tokens * elt;
    if ((uint64_t) (end - p) != n_hidden) {
        fail(PIPE_ERR_BAD_FRAME,
             "pipe: FWD_RESP hidden bytes %lld, want %lld",
             (long long) (end - p), (long long) n_hidden);
    }
    r.hidden.assign(p, end);
    return r;
}

// ---------------------------------------------------------------------------
// TOKEN
//
//   u32  n_tokens
//   i32  token_ids[n_tokens]

std::vector<uint8_t> pipe_encode_token(const pipe_token & p) {
    const uint64_t total = 4 + (uint64_t) p.token_ids.size() * 4;
    if (total > PIPE_MAX_PAYLOAD) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: TOKEN encode size %llu exceeds max payload",
             (unsigned long long) total);
    }
    std::vector<uint8_t> out((size_t) total);
    uint8_t * w = out.data();
    wr_u32(w, (uint32_t) p.token_ids.size());
    for (int32_t v : p.token_ids) { wr_i32(w, v); }
    return out;
}

pipe_token pipe_decode_token(const uint8_t * buf, size_t len) {
    if (len < 4) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: TOKEN payload %zu bytes too small", len);
    }
    const uint8_t * p   = buf;
    const uint8_t * end = buf + len;
    pipe_token r;
    const uint32_t n = rd_u32(p);
    if ((uint64_t) (end - p) != (uint64_t) n * 4) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: TOKEN ids bytes %lld, want %lld",
             (long long) (end - p), (long long) n * 4);
    }
    r.token_ids.reserve(n);
    for (uint32_t i = 0; i < n; ++i) { r.token_ids.push_back(rd_i32(p)); }
    return r;
}

// ---------------------------------------------------------------------------
// ERROR
//
//   u32  code
//   u16  msg_len
//   u8[] msg

std::vector<uint8_t> pipe_encode_error(const pipe_error & p) {
    if (p.msg.size() > 0xffff) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: ERROR msg %zu bytes exceeds u16", p.msg.size());
    }
    std::vector<uint8_t> out(4 + 2 + p.msg.size());
    uint8_t * w = out.data();
    wr_u32(w, p.code);
    // u16 msg_len, little-endian
    w[0] = (uint8_t) (p.msg.size() & 0xff);
    w[1] = (uint8_t) ((p.msg.size() >> 8) & 0xff);
    w += 2;
    if (!p.msg.empty()) {
        std::memcpy(w, p.msg.data(), p.msg.size());
    }
    return out;
}

pipe_error pipe_decode_error(const uint8_t * buf, size_t len) {
    if (len < 6) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: ERROR payload %zu bytes too small", len);
    }
    const uint8_t * p   = buf;
    const uint8_t * end = buf + len;
    pipe_error r;
    r.code = rd_u32(p);
    const uint32_t msg_len = (uint32_t) p[0] | ((uint32_t) p[1] << 8);
    p += 2;
    if ((uint64_t) (end - p) != msg_len) {
        fail(PIPE_ERR_BAD_FRAME, "pipe: ERROR msg bytes %lld, want %u",
             (long long) (end - p), msg_len);
    }
    r.msg.assign((const char *) p, msg_len);
    return r;
}

// ---------------------------------------------------------------------------
// HELLO validation

void pipe_validate_hello(const pipe_hello & peer,
                         int32_t our_n_layer, int32_t our_n_embd, int32_t our_hidden_type,
                         const std::vector<llama_pipeline_stage> & stages) {
    if (peer.n_layer != our_n_layer) {
        fail(PIPE_ERR_HELLO,
             "pipe: HELLO n_layer mismatch: peer=%d ours=%d (stages of different models?)",
             peer.n_layer, our_n_layer);
    }
    if (peer.n_embd != our_n_embd) {
        fail(PIPE_ERR_HELLO,
             "pipe: HELLO n_embd mismatch: peer=%d ours=%d",
             peer.n_embd, our_n_embd);
    }
    if (peer.hidden_type != our_hidden_type) {
        fail(PIPE_ERR_HELLO,
             "pipe: HELLO hidden_type mismatch: peer=%d ours=%d (one negotiated value per pipeline)",
             peer.hidden_type, our_hidden_type);
    }
    // reject an unknown hidden_type before anyone sizes a tensor from it
    pipe_hidden_elt_size(peer.hidden_type);

    // the peer's declared band must be one of the stages we believe in
    bool found = false;
    for (const auto & s : stages) {
        if (s.first == peer.layer_first && s.last == peer.layer_last) {
            found = true;
            break;
        }
    }
    if (!found) {
        fail(PIPE_ERR_HELLO,
             "pipe: HELLO peer band [%d, %d] not in the agreed stage set",
             peer.layer_first, peer.layer_last);
    }

    // the combined stage set must cover [0, n_layer-1] exactly once.
    // llama_pipeline_validate_stages throws std::runtime_error; wrap it as a
    // HELLO error so the caller sends PIPE_ERROR and closes.
    try {
        llama_pipeline_validate_stages(stages, our_n_layer);
    } catch (const std::runtime_error & e) {
        fail(PIPE_ERR_HELLO, "pipe: HELLO stage-set inconsistent: %s", e.what());
    }
}

// ---------------------------------------------------------------------------
// connection-level frame IO

bool pipe_send_frame(pipe_socket_t & sock, pipe_frame_type type, uint64_t seq_id,
                     const uint8_t * payload, size_t payload_len) {
    pipe_frame_header h;
    h.magic   = PIPE_MAGIC;
    h.version = PIPE_VERSION;
    h.type    = (uint32_t) type;
    h.flags   = 0;
    h.seq_id  = seq_id;
    h.length  = (uint64_t) payload_len;

    uint8_t hdr[PIPE_HEADER_SIZE];
    pipe_encode_header(hdr, h);
    if (!sock.send_data(hdr, PIPE_HEADER_SIZE)) {
        return false;
    }
    if (payload_len > 0 && !sock.send_data(payload, payload_len)) {
        return false;
    }
    return true;
}

bool pipe_recv_frame(pipe_socket_t & sock, pipe_frame_type & type, uint64_t & seq_id,
                     std::vector<uint8_t> & payload) {
    uint8_t hdr[PIPE_HEADER_SIZE];
    if (!sock.recv_data(hdr, PIPE_HEADER_SIZE)) {
        return false;
    }
    // throws pipe_protocol_error on bad magic/version/oversized length, before
    // any allocation is made from the untrusted length field
    const pipe_frame_header h = pipe_decode_header(hdr);

    payload.resize((size_t) h.length);
    if (h.length > 0 && !sock.recv_data(payload.data(), (size_t) h.length)) {
        return false;
    }
    type   = (pipe_frame_type) h.type;
    seq_id = h.seq_id;
    return true;
}

bool pipe_send_error(pipe_socket_t & sock, uint64_t seq_id, pipe_error_code code,
                     const std::string & msg) {
    pipe_error e;
    e.code = (uint32_t) code;
    e.msg  = msg;
    std::vector<uint8_t> payload;
    try {
        payload = pipe_encode_error(e);
    } catch (const pipe_protocol_error &) {
        return false;
    }
    return pipe_send_frame(sock, PIPE_ERROR, seq_id, payload.data(), payload.size());
}
