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
// 3 -> 4 (2026-08-05): request activations are f32, not f16. A stale binary would
// mis-decode a 2-byte stream as 4-byte values, so this MUST reject rather than
// reinterpret. Same reasoning as the 1 -> 2 bump on the partial-return path.
// 4 -> 5 (2026-08-05): PIPE_EXPERT_PREFETCH_HINT. This bump is NOT about decoding
// -- the hint is a new frame TYPE and adds no field to any existing payload, so a
// stale worker could never mis-read one. It is about the frame dispatch loop:
// wp-expert-worker.cpp answers any frame that is not PIPE_EXPERT_DISPATCH_REQ with
// PIPE_ERROR and CLOSES THE CONNECTION. A pre-hint worker would therefore not
// ignore the hint, it would kill the session on the first one -- mid-run, after a
// successful HELLO, looking exactly like a worker crash. Rejecting at HELLO turns
// that into one clear line at startup.
// 6: the prefetch hint carries PROVENANCE. A hint derived from a token the
//    target will certainly process (id_last) and one derived from a PREDICTION
//    cost the same to send and are worth very different amounts to keep, and a
//    worker that cannot tell them apart has to lease them identically. Measured:
//    predicted hints displaced ~200 ground-truth pages precisely because they
//    held slots on equal terms.
static constexpr uint32_t PIPE_VERSION = 6u;

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
    PIPE_EXPERT_PREFETCH_HINT = 11,
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
//   f32 activations[n_tokens * n_embd]
struct pipe_expert_dispatch_req {
    int32_t                             layer    = -1;
    uint32_t                            n_tokens = 0;
    std::vector<pipe_expert_assignment> assignments;
    // *** f32, NOT f16. CHANGED 2026-08-05 -- THIS WAS A CORRECTNESS BUG. ***
    // The 2026-08-04 fix repaired only the RETURN direction (pipe_expert_partial).
    // The REQUEST direction stayed f16, so every routed expert RECEIVED f16(x)
    // (~3e-4 relative) while attention, the shared expert and the residual all saw
    // f32 x -- an asymmetric, deterministic perturbation on every single layer.
    // WP_SELFCHECK could never see it: gather and dense both consume the SAME
    // already-rounded activation, so the probe compares two equally-wrong paths.
    // Costs 2x bytes on the request path; decode is page-in bound, so measure
    // rather than assume that costs anything.
    std::vector<float>                  activations;

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

// A prefetch hint: "you are about to be asked for these experts on this layer".
//
// FIRE AND FORGET. There is no response frame and no seq_id correlation; the
// header's seq_id is informational only. A worker is free to ignore the hint
// entirely, act on part of it, or be interrupted mid-warm by a real request --
// the hint carries NO correctness weight. That is deliberate and it is the
// property that makes this safe to add: the dispatch path is unchanged, so a
// hint that is wrong, late, or dropped can only cost I/O, never an answer.
//
// EXPERT IDS, NOT PAGES. The spine cannot resolve a page: under cross-machine
// dispatch deepseek4.cpp marks the routed experts TENSOR_SKIP, so the spine has
// no routed-expert catalog to resolve against (and in the config of record it has
// no WeightPager at all -- llama-server runs the dense spine without
// --weight-paging, so llama_wp_on_draft_tokens returns 0 at its null check before
// doing anything). Each worker owns the shard, so each worker does its own
// expert -> page resolution against its own catalog. This is why the hint stops
// at expert ids: it is the last representation both sides can agree on.
//
// The ids are ascending and unique on the wire. Ascending is not cosmetic --
// it is the ORDER the reads should be issued in, which is the whole point on the
// prefill path (a sorted stream is the only regime where the drive's sequential
// bandwidth is reachable), and it lets the decoder validate dedup in one pass
// instead of building a set.
//
// Payload:
//   i32 layer
//   u32 n_experts
//   u32 provenance                (PIPE_HINT_*)
//   i32 expert_id[n_experts]      (strictly ascending, all >= 0)
//
// PROVENANCE IS NOT A HINT ABOUT QUALITY, IT IS A STATEMENT ABOUT CERTAINTY.
// CERTAIN ids come from a token the target is already committed to processing --
// they cannot be wrong. PREDICTED ids come from the previous draft block, which
// is right about 40% of the time per expert. The worker keeps both, but it must
// not spend the same residency on them: a predicted page that outranks a certain
// one converts a free guess into a displaced fact.
enum pipe_hint_provenance : uint32_t {
    PIPE_HINT_CERTAIN   = 0,
    PIPE_HINT_PREDICTED = 1,
};

struct pipe_expert_prefetch_hint {
    int32_t              layer      = -1;
    uint32_t             provenance = PIPE_HINT_CERTAIN;
    std::vector<int32_t> expert_ids;
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
std::vector<uint8_t> pipe_encode_expert_prefetch_hint(const pipe_expert_prefetch_hint & p);
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
pipe_expert_prefetch_hint pipe_decode_expert_prefetch_hint(
    const uint8_t * buf, size_t len);
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
