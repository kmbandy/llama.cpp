#pragma once

// Phase 2 cross-machine pipeline protocol: frame encode/decode.
//
// Implements docs/dev/2026-07-28-pipeline-protocol-design.md. Length-prefixed
// frames over TCP, all integers little-endian, 24-byte fixed header:
//
//   u32  magic   = 0x4C4C5050 ("LLPP")
//   u32  version = 1
//   u32  type    (pipe_frame_type)
//   u32  flags   (reserved, 0)
//   u64  seq_id
//   u64  length  (payload bytes following the header)
//   u8[] payload
//
// Encode/decode is byte-explicit (read_le32/write_le32 style), never a memcpy
// of a struct -- header layout is fixed on the wire and independent of host
// struct padding or endianness.
//
// HELLO is the consistency gate: on connect both sides exchange HELLO and
// validate the combined stage set with llama_pipeline_validate_stages(). Any
// inconsistency (gap, overlap, n_layer/n_embd/hidden_type mismatch) is
// rejected with PIPE_ERROR and close. A pipeline assembled from an
// inconsistent stage set must fail at connect, never produce output.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama-pipeline.h"

// ---------------------------------------------------------------------------
// wire constants

static constexpr uint32_t PIPE_MAGIC   = 0x4C4C5050u; // "LLPP"
// VERSION 2 (2026-08-04): expert partials are f32 on the wire, not f16. The bump is
// deliberate -- a spine and a worker built either side of this change MUST refuse each
// other loudly rather than silently mis-decode a 2-byte stream as 4-byte values.
// 2 -> 3 (2026-08-05): the dispatch request carries swiglu_clamp. A worker built
// before this change applies NO clamp, which is a silent correctness bug (see
// pipe_expert_dispatch_req below), so the version must reject it rather than let
// it compute an unclamped SwiGLU that looks perfectly healthy on the wire.
static constexpr uint32_t PIPE_VERSION = 3u;

// NOTE: the design doc says "24-byte fixed header" but its own field list
// (4x u32 + u64 seq_id + u64 length = 16 + 8 + 8) sums to 32 bytes. The field
// list is the contract -- magic/version/type/flags/seq_id/length are all
// required -- so the header is 32 bytes on the wire. The "24" is an arithmetic
// slip in the doc; flagged in the implementation report.
static constexpr size_t   PIPE_HEADER_SIZE = 32;

// Hard cap on a frame payload. A FWD_REQ is n_ubatch * n_embd * 4 bytes of
// hidden plus position arrays; even 8192 tokens * 8192 embd * 4 = 256 MiB.
// A peer claiming more than this is rejected before any allocation.
static constexpr uint64_t PIPE_MAX_PAYLOAD = 512ull * 1024ull * 1024ull; // 512 MiB

enum pipe_frame_type : uint32_t {
    PIPE_HELLO               = 1,
    PIPE_FWD_REQ             = 2,
    PIPE_FWD_RESP            = 3,
    PIPE_TOKEN               = 4,
    PIPE_ERROR               = 5,
    PIPE_PING                = 6,
    PIPE_PONG                = 7,
    PIPE_EXPERT_DISPATCH_REQ = 8,
    PIPE_EXPERT_PARTIAL      = 9,
    PIPE_EXPERT_HELLO_ACK    = 10,
};

enum pipe_role : uint32_t {
    PIPE_ROLE_HEAD   = 0,
    PIPE_ROLE_MIDDLE = 1,
    PIPE_ROLE_TAIL   = 2,
};

enum pipe_hidden_type : int32_t {
    PIPE_HIDDEN_F32 = 0,
    PIPE_HIDDEN_F16 = 1,
};

enum pipe_error_code : uint32_t {
    PIPE_ERR_GENERIC       = 0,
    PIPE_ERR_HELLO         = 1, // handshake / stage-set inconsistency
    PIPE_ERR_BAD_FRAME     = 2, // malformed frame (magic/version/length)
    PIPE_ERR_STALE_SEQ     = 3, // seq_id from a dead session
    PIPE_ERR_DECODE        = 4, // llama_decode failed on a stage
    PIPE_ERR_EXPERT_RANGE  = 5, // requested expert is not owned by this worker
    PIPE_ERR_EXPERT_LAYER  = 6, // requested layer is not served by this worker
    PIPE_ERR_EXPERT_COMPUTE = 7,
};

// ---------------------------------------------------------------------------
// parsed payloads

struct pipe_hello {
    uint32_t role        = 0;
    int32_t  layer_first = -1;
    int32_t  layer_last  = -1;
    int32_t  n_layer     = 0;
    int32_t  n_embd      = 0;
    int32_t  hidden_type = PIPE_HIDDEN_F32;
    uint32_t model_hash  = 0;

    static constexpr size_t WIRE_SIZE = 6 * 4 + 4; // 28 bytes
};

struct pipe_fwd_req {
    uint32_t              n_tokens = 0;
    uint32_t              n_pos_per_embd = 1;
    std::vector<int32_t>  pos;         // n_tokens * n_pos_per_embd
    // Phase 2: n_seqs fixed to 1; the field is reserved on the wire.
    std::vector<int32_t>  seq_tokens;  // n_seqs entries, sums to n_tokens
    std::vector<uint8_t>  hidden;      // n_embd * n_tokens * elt_size(hidden_type)
};

struct pipe_fwd_resp {
    uint32_t             n_tokens = 0;
    std::vector<uint8_t> hidden;   // n_embd * n_tokens * elt_size
};

struct pipe_token {
    std::vector<int32_t> token_ids; // n_tokens sampled ids
};

struct pipe_error {
    uint32_t    code = 0;
    std::string msg;
};

enum pipe_expert_role : uint32_t {
    PIPE_EXPERT_ROLE_CLIENT = 0,
    PIPE_EXPERT_ROLE_WORKER = 1,
};

// Expert endpoints still exchange PIPE_HELLO frames. The expert payload is
// separate from pipe_hello so the existing stage protocol remains byte-for-byte
// compatible. A worker advertises the exact layer set and inclusive expert
// range it can serve. model_identity identifies the logical source model and
// is common to all shards made from it. shard_identity is the shard manifest
// content hash. The client sends the same identities and hparams with an empty
// layer set.
struct pipe_expert_hello {
    uint32_t             role          = PIPE_EXPERT_ROLE_CLIENT;
    int32_t              hidden_type   = PIPE_HIDDEN_F16;
    int32_t              n_embd        = 0;
    int32_t              n_ff_exp      = 0;
    int32_t              n_expert      = 0;
    int32_t              n_expert_used = 0;
    int32_t              expert_first  = -1;
    int32_t              expert_last   = -1;
    uint32_t             n_slots       = 0;
    std::vector<int32_t> layers;
    std::string          model_identity;
    std::string          shard_identity;
};

struct pipe_expert_hello_ack {
    bool        accepted = false;
    std::string reason;
};

struct pipe_expert_assignment {
    int32_t            expert_id = -1;
    std::vector<float> weights; // one final router weight per token
};

// seq_id in the fixed frame header is the request's step/sequence id.
//
// Payload:
//   i32 layer
//   u32 n_tokens
//   u32 n_assignments
//   f32 swiglu_clamp
//   repeated { i32 expert_id; f32 weights[n_tokens] }
//   f16 activations[n_tokens * n_embd]
struct pipe_expert_dispatch_req {
    int32_t                             layer    = -1;
    uint32_t                            n_tokens = 0;
    std::vector<pipe_expert_assignment> assignments;
    std::vector<uint16_t>               activations;

    // *** ADDED 2026-08-05 -- THIS WAS A CORRECTNESS BUG. ***
    // hparams.swiglu_clamp_exp[layer] for this layer; <= 0 means "no clamp".
    // DS4-Flash-0731 ships deepseek4.swiglu_clamp_exp = 10.0, and in the
    // DISTRIBUTED configuration that clamp was applied NOWHERE:
    //   1. llama-graph.cpp build_moe_ffn() returns at the `expert_dispatch != nullptr`
    //      branch BEFORE reaching the clamped SwiGLU switch, so the spine never
    //      applies it -- the clamp is dead code whenever --expert-dispatch is on.
    //   2. wp-expert-worker.cpp computed a bare ggml_swiglu_split(gate, up).
    //   3. There was no field on the wire, so the limit could not even REACH a worker.
    // Net effect: every routed expert on every layer computed silu(g)*u instead of
    // silu(min(g, limit)) * clamp(u, -limit, limit). Deterministic, backend- and
    // width-independent, and invisible to WP_SELFCHECK (which compares gather vs
    // dense -- both arms unclamped) and to WP_EXPERT_GATHER=0 (which only changes
    // the graph shape INSIDE a worker, not the FFN nonlinearity).
    // The shared expert is computed on the spine and goes through the normal
    // build_ffn path, so swiglu_clamp_shexp was never affected.
    float                               swiglu_clamp = 0.0f;
};

// Payload:
//   i32 layer
//   u32 n_tokens
//   f16 partial[n_tokens * n_embd]
struct pipe_expert_partial {
    int32_t            layer    = -1;
    uint32_t           n_tokens = 0;
    // *** f32, NOT f16. CHANGED 2026-08-04 -- THIS WAS A CORRECTNESS BUG. ***
    // Each worker sums ITS OWN subset of a layer's experts in f32 and the spine adds
    // the per-worker subtotals. Sending the subtotal as f16 rounded it to an 11-bit
    // mantissa (~5e-4 relative) AT THE PARTITION BOUNDARY, so the final MoE output
    // depended on WHICH WORKER GOT WHICH EXPERT -- and that assignment is chosen by
    // choose_worker() from residency, in-request assigned counts, and a rotating
    // machine cursor, all of which move with batch width. Net effect: changing the
    // speculative draft length (i.e. conf_min) silently changed the generated text at
    // temperature 0, because moving one expert between workers re-rounded two
    // subtotals and perturbed the layer output by ~1e-3 relative -- four orders of
    // magnitude above f32 reordering noise -- which the hyper-connection gates and the
    // discontinuous router top-k then amplified into a different trajectory.
    // f32 removes the amplifier entirely; what remains is ordinary f32 reordering
    // (~1e-7). Costs 2x bytes on the partial-return path only.
    std::vector<float> partial;
};

// ---------------------------------------------------------------------------
// header + frame primitives (byte-explicit, little-endian)

struct pipe_frame_header {
    uint32_t magic   = 0;
    uint32_t version = 0;
    uint32_t type    = 0;
    uint32_t flags   = 0;
    uint64_t seq_id  = 0;
    uint64_t length  = 0;
};

// Serialise `h` into exactly PIPE_HEADER_SIZE bytes (little-endian).
void pipe_encode_header(uint8_t out[PIPE_HEADER_SIZE], const pipe_frame_header & h);

// Parse a header from exactly PIPE_HEADER_SIZE bytes. Throws
// pipe_protocol_error on bad magic, wrong version, or length >
// PIPE_MAX_PAYLOAD. Does NOT allocate.
pipe_frame_header pipe_decode_header(const uint8_t in[PIPE_HEADER_SIZE]);

// Thrown on any framing/validation error. Carries the PIPE_ERROR code the
// peer should be sent when the error surfaces on a live connection.
struct pipe_protocol_error : std::runtime_error {
    explicit pipe_protocol_error(pipe_error_code code, const std::string & m)
        : std::runtime_error(m), code(code) {}
    pipe_protocol_error(pipe_error_code code, const char * m)
        : std::runtime_error(m), code(code) {}
    pipe_error_code code;
};

// ---------------------------------------------------------------------------
// payload encode/decode (pure buffers; no sockets here)

std::vector<uint8_t> pipe_encode_hello     (const pipe_hello    & p);
std::vector<uint8_t> pipe_encode_fwd_req   (const pipe_fwd_req  & p, int32_t hidden_type);
std::vector<uint8_t> pipe_encode_fwd_resp  (const pipe_fwd_resp & p);
std::vector<uint8_t> pipe_encode_token     (const pipe_token    & p);
std::vector<uint8_t> pipe_encode_error     (const pipe_error    & p);
std::vector<uint8_t> pipe_encode_expert_hello(const pipe_expert_hello & p);
std::vector<uint8_t> pipe_encode_expert_hello_ack(const pipe_expert_hello_ack & p);
std::vector<uint8_t> pipe_encode_expert_dispatch_req(const pipe_expert_dispatch_req & p);
std::vector<uint8_t> pipe_encode_expert_partial(const pipe_expert_partial & p);
// PING/PONG carry no payload.

pipe_hello     pipe_decode_hello     (const uint8_t * buf, size_t len);
pipe_fwd_req   pipe_decode_fwd_req   (const uint8_t * buf, size_t len, int32_t n_embd, int32_t hidden_type);
pipe_fwd_resp  pipe_decode_fwd_resp  (const uint8_t * buf, size_t len, int32_t n_embd, int32_t hidden_type);
pipe_token     pipe_decode_token     (const uint8_t * buf, size_t len);
pipe_error     pipe_decode_error     (const uint8_t * buf, size_t len);
pipe_expert_hello pipe_decode_expert_hello(const uint8_t * buf, size_t len);
pipe_expert_hello_ack pipe_decode_expert_hello_ack(const uint8_t * buf, size_t len);
pipe_expert_dispatch_req pipe_decode_expert_dispatch_req(
    const uint8_t * buf, size_t len, int32_t n_embd);
pipe_expert_partial pipe_decode_expert_partial(
    const uint8_t * buf, size_t len, int32_t n_embd);

// ---------------------------------------------------------------------------
// HELLO validation

// Validate a peer's HELLO against our own identity and the full stage set we
// believe the pipeline has. `stages` must be the complete ordered set
// (head .. tail) assembled from every connected peer plus our own band, and
// is checked with llama_pipeline_validate_stages(). Also enforces the
// cross-machine invariants a single process cannot check alone:
//   - bands contiguous, non-overlapping, covering [0, n_layer-1]  (via helper)
//   - peer.n_layer == ours, peer.n_embd == ours
//   - peer.hidden_type == ours (one negotiated value per pipeline)
//   - the peer's declared band actually appears in `stages`
// Throws pipe_protocol_error(PIPE_ERR_HELLO, ...) on any mismatch.
void pipe_validate_hello(const pipe_hello & peer,
                         int32_t our_n_layer, int32_t our_n_embd, int32_t our_hidden_type,
                         const std::vector<llama_pipeline_stage> & stages);

// element size in bytes for a hidden_type (4 for F32, 2 for F16). Throws on
// an unknown type.
uint32_t pipe_hidden_elt_size(int32_t hidden_type);

// ---------------------------------------------------------------------------
// connection-level frame IO (needs pipe-transport; declared here so callers
// have one header for the whole protocol)

struct pipe_socket_t;

// Send one frame (header + payload) on `sock`. Returns false on transport
// error. `payload` may be empty (PING/PONG).
bool pipe_send_frame(pipe_socket_t & sock, pipe_frame_type type, uint64_t seq_id,
                     const uint8_t * payload, size_t payload_len);

// Receive one frame. On success fills `type`, `seq_id`, and `payload`
// (resized to the frame length; empty for PING/PONG). Returns false on
// transport error. Throws pipe_protocol_error on a malformed frame (bad
// magic/version/oversized length) -- the caller should send PIPE_ERROR and
// close. Never allocates before the header's length field is validated.
bool pipe_recv_frame(pipe_socket_t & sock, pipe_frame_type & type, uint64_t & seq_id,
                     std::vector<uint8_t> & payload);

// Convenience: send a PIPE_ERROR frame. Best-effort; returns false if the
// transport is already broken.
bool pipe_send_error(pipe_socket_t & sock, uint64_t seq_id, pipe_error_code code,
                     const std::string & msg);
