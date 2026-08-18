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
// 5 -> 6 (2026-08-08): expert dispatch may be split into BEGIN and ACTS frames.
// PIPE_EXPERT_DISPATCH_BEGIN carries the layer, token count, assignments and
// SwiGLU clamp; PIPE_EXPERT_DISPATCH_ACTS carries only f32 activations. Version
// 6 workers accept both this pair and the monolithic request for A/B compatibility.
// The prefetch hint also carries PROVENANCE. A hint derived from a token the
//    target will certainly process (id_last) and one derived from a PREDICTION
//    cost the same to send and are worth very different amounts to keep, and a
//    worker that cannot tell them apart has to lease them identically. Measured:
//    predicted hints displaced ~200 ground-truth pages precisely because they
//    held slots on equal terms.
// 6 -> 7 (2026-08-15): dense-segment frames. The segment HELLO, forward and
// control payloads have their own versions, but a pre-7 peer does not know
// their frame types and must fail during the connection handshake.
// 7 -> 8 (2026-08-15): the dense-segment HELLO and response now carry the
// vocabulary/output widths. A terminal response is logits, not hidden state,
// so using n_embd there would silently corrupt the server sampler.
// 8 -> 9 (2026-08-15): HELLO ACK reports recurrent snapshot depth so the
// head can clamp a speculative draft before it asks a segment to roll back.
// 9 -> 10 (2026-08-15): LOGITS-ON-HEAD. The terminal segment may now return the
// post-output_norm HIDDEN STATE (n_embd) instead of LOGITS (n_vocab), leaving the
// LM head projection to the head. That is 5120 vs 248320 f32 per token -- a 48x
// cut on the terminal leg, ~993 KB -> ~20 KB, worth ~9 ms/token of wire at the
// measured ~888 Mbps. The two payloads are BOTH f32 arrays of [n_tokens, W] with
// nothing self-describing about W, so a head and a tail that disagree about the
// kind do not fail -- they mis-slice the same bytes and the head samples from a
// buffer of hidden states. That is a silent, catastrophic correctness failure, so
// it MUST be rejected at the handshake. The negotiation lives in the segment HELLO
// (below) and this framing bump makes a pre-10 peer fail even earlier.
// 10 -> 11 (2026-08-16): INTERIOR TAPS. A segment may now be asked to extract the input
// hidden state of one or more layers INSIDE its band and return them alongside the
// forward response, so that a speculative draft running on the head can condition on
// target activations the head does not compute itself. A DFlash/DSpark draft taps fixed
// target layers (dflash.target_layers, e.g. [5,17,29,41,53]); with a 0-48 head those
// last taps live on another machine.
//
// This is version-gated for the same reason terminal_kind was, and the failure is even
// quieter. If a head expects taps and a peer does not send them, the head reads a stale
// or zero buffer as if it were the tap. Nothing mis-sizes and nothing crashes: the
// drafter simply conditions on garbage and proposes worse tokens. Speculative decoding
// is exact by construction -- the target verifies every draft token -- so the OUTPUT
// STAYS BIT-IDENTICAL and even a temp-0 parity test passes. The only symptom is a
// degraded acceptance rate, which is indistinguishable from an ordinary bad draft model.
// That is strictly harder to detect than the terminal_kind mis-slice, so it must be
// rejected at the handshake and the framing bump makes a pre-11 peer fail earlier still.
// 11 -> 12 (2026-08-16): NEXTN WIRE DEDUP. The terminal segment used to ship the
// nextn sideband on EVERY forward response, unconditionally, because nextn_width was
// a load-time property of the worker and was never negotiated. Measured on the
// dense-segment tail under the HIDDEN terminal payload: 2 x 40960 bytes per token at
// n_embd=5120 -- the post-output_norm hidden state travelled twice, once as
// `activations` and once as `nextn`, and on every arm except draft-mtp the second copy
// was read by nobody (server-context.cpp gates the consume on
// common_speculative_need_embd_nextn(), which only draft-mtp answers true).
//
// Version 12 negotiates it: the segment HELLO carries nextn_need, the ACK answers with
// the nextn_width the segment WILL send (0 when the head declared no need), and the
// forward response carries nextn_aliased so a tail whose nextn is bit-identical to the
// terminal hidden state can ship it ONCE and let the head reconstruct.
//
// This MUST be version-gated. The response encodes nextn as a bare f32 run whose length
// is implied by nextn_width and nextn_aliased; a pre-12 tail sends the run
// unconditionally and knows no aliasing flag, so a v12 head that negotiated
// nextn_width=0 would read the tail's 40960 extra bytes per token as part of the frame
// and fail the length check -- loudly, but only after a successful HELLO, which looks
// exactly like a worker crash mid-run. Worse in the other direction: a pre-12 HEAD
// cannot express need, so a v12 tail would default it to "no need" and silently starve
// draft-mtp of the sideband it verifies against. Rejecting at the handshake turns both
// into one clear line at startup.
// 12 -> 13 (2026-08-17): PARTIAL DTYPE TAG. pipe_expert_partial now carries an
// explicit `dtype` field (pipe_hidden_type: F32=0 default, F16=1) ahead of the
// partial array, so the SPINE decodes whatever a worker actually sent rather
// than assuming f32. This is what makes WP_EXPERT_PARTIAL_DTYPE=f16 (worker-side,
// per-process, default OFF -- see wp-expert-worker.cpp) safe to enable on some
// workers and not others: the tag is self-describing, so a spine that has never
// heard of the env var still decodes correctly, and a worker/spine version
// mismatch cannot silently reinterpret one dtype's bytes as the other's.
//
// This MUST still be version-gated even though the tag is self-describing,
// because the tag itself is a NEW field: a pre-13 peer's decoder expects the
// partial array to start immediately after (layer, n_tokens) and has no idea a
// dtype word was inserted. Without the bump, a v13 worker talking to a pre-13
// spine would have its `dtype` word (0 for the default f32 case) misread as the
// first 4 bytes of the partial's row 0 -- a silent off-by-4-bytes corruption of
// exactly the kind this file's whole version history exists to prevent. The
// bump makes that pairing fail loudly at HELLO instead. Once both peers are on
// version 13, the tag alone is what keeps a WORKER-side config knob from ever
// being able to corrupt the SPINE's sum: the spine does not need to know or
// agree with WP_EXPERT_PARTIAL_DTYPE, it only needs to trust the tag on the
// frame it just received, which it always can because the tag was written by
// the same process that wrote the bytes after it.
static constexpr uint32_t PIPE_VERSION = 13u;

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
    PIPE_EXPERT_DISPATCH_BEGIN = 12,
    PIPE_EXPERT_DISPATCH_ACTS  = 13,
    PIPE_SEGMENT_HELLO         = 14,
    PIPE_SEGMENT_HELLO_ACK     = 15,
    PIPE_SEGMENT_FWD_REQ       = 16,
    PIPE_SEGMENT_FWD_RESP      = 17,
    PIPE_SEGMENT_CTRL          = 18,
    PIPE_SEGMENT_CTRL_ACK      = 19,
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

enum pipe_segment_wire_precision : uint32_t {
    PIPE_SEGMENT_WIRE_F32 = 1,
};

// What the TERMINAL (tail) segment puts in pipe_segment_fwd_resp::activations.
// Non-terminal segments always return hidden state and ignore this entirely.
enum pipe_segment_terminal_kind : uint32_t {
    // Legacy: the tail runs output_norm AND the LM head and returns
    // [n_tokens, n_vocab] logits. Kept for A/B (WP_SEGMENT_TAIL_LOGITS=1).
    PIPE_SEGMENT_TERMINAL_LOGITS = 1,
    // Logits-on-head: the tail runs output_norm and STOPS, returning
    // [n_tokens, n_embd] normed hidden state. The head does the projection.
    PIPE_SEGMENT_TERMINAL_HIDDEN = 2,
};

enum pipe_segment_capability : uint64_t {
    PIPE_SEGMENT_CAP_FWD          = 1ull << 0,
    PIPE_SEGMENT_CAP_RESET        = 1ull << 1,
    PIPE_SEGMENT_CAP_KV_TRIM      = 1ull << 2,
    PIPE_SEGMENT_CAP_PROMPT_REUSE = 1ull << 3,
    PIPE_SEGMENT_CAP_RECURRENT    = 1ull << 4,
};

enum pipe_segment_ctrl_type : uint32_t {
    PIPE_SEGMENT_CTRL_RESET        = 1,
    PIPE_SEGMENT_CTRL_KV_TRIM      = 2,
    PIPE_SEGMENT_CTRL_PROMPT_REUSE = 3,
};

enum pipe_segment_ctrl_status : uint32_t {
    PIPE_SEGMENT_CTRL_APPLIED = 1,
    PIPE_SEGMENT_CTRL_MISS    = 2,
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

// Split dispatch: the BEGIN payload is the dispatch metadata without the
// activation tensor. The ACTS payload is the f32 activation tensor only.
struct pipe_expert_dispatch_begin {
    int32_t                            layer = -1;
    uint32_t                           n_tokens = 0;
    std::vector<pipe_expert_assignment> assignments;
    float                              swiglu_clamp = 0.0f;
};

struct pipe_expert_dispatch_acts {
    std::vector<float> activations;
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
//   i32 dtype                        (pipe_hidden_type: F32=0 default, F16=1)
//   u8[] partial   n_tokens*n_embd elements, elt_size(dtype) bytes each,
//                  f32 or f16 per IEEE754 depending on dtype
struct pipe_expert_partial {
    int32_t            layer    = -1;
    uint32_t           n_tokens = 0;
    // *** f32 ON THE WIRE BY DEFAULT. CHANGED 2026-08-04 -- THIS WAS A CORRECTNESS
    // BUG (see the full history below); f32 is the safe default and stays that way
    // unless a worker opts in to f16 via WP_EXPERT_PARTIAL_DTYPE=f16.
    //
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
    //
    // *** dtype ADDED 2026-08-17 (WP_EXPERT_PARTIAL_DTYPE). ***
    // The correctness bug above was about f16 being UNCONDITIONAL and UN-TAGGED: the
    // spine had no way to know a partial had been rounded, so a partition-dependent
    // subtotal error looked like ordinary noise. This is different: it is an OPT-IN,
    // PER-WORKER, SELF-DESCRIBING encoding. `dtype` says exactly what `partial`'s
    // bytes mean on THIS frame, so the spine always decodes correctly regardless of
    // what any worker's env var says -- a worker/spine version or config mismatch can
    // misinterpret nothing, because the tag travels with the bytes it describes.
    // What f16 partials reintroduce is the SAME quantization risk the 2026-08-04 note
    // describes (~5e-4 relative per worker, amplified by hyper-connection gates and
    // top-k), which is why it is default OFF and meant for links that are bandwidth-
    // bound (e.g. a 1 GbE hop to a remote worker) rather than for every deployment.
    // Acceptable under f16 KV; risky under turbo4 KV, per the same amplification
    // mechanism documented above -- that is why this stays opt-in rather than becoming
    // the new default.
    int32_t             dtype = PIPE_HIDDEN_F32;
    std::vector<float> partial;
};

// ---------------------------------------------------------------------------
// dense-segment payloads

// The fixed framing version is PIPE_VERSION. Each segment payload starts with
// its own version so fields can evolve without ambiguity once the segment
// family is deployed.
// HELLO 3 -> 4 (2026-08-15): terminal_kind negotiation (logits-on-head).
// HELLO 4 -> 5 (2026-08-16): interior tap negotiation (tap_layers / tap_width).
// HELLO 5 -> 6 (2026-08-16): nextn sideband negotiation (nextn_need / nextn_width).
static constexpr uint32_t PIPE_SEGMENT_HELLO_VERSION    = 6;
// FWD 2 -> 3 (2026-08-16): the response may carry interior tap rows after `nextn`.
// FWD 3 -> 4 (2026-08-16): nextn_aliased. When set, the `nextn` run is OMITTED from the
// payload and is bit-identical to `activations`; the head reconstructs it locally.
static constexpr uint32_t PIPE_SEGMENT_FWD_VERSION      = 4;
static constexpr uint32_t PIPE_SEGMENT_CTRL_VERSION     = 1;
static constexpr uint32_t PIPE_SEGMENT_CTRL_ACK_VERSION = 1;

struct pipe_segment_hello {
    uint32_t    version                 = PIPE_SEGMENT_HELLO_VERSION;
    uint32_t    segment_id              = 0;
    int32_t     layer_first              = -1;
    int32_t     layer_last               = -1;
    std::string model_identity_sha256;
    int32_t     n_embd                  = 0;
    uint32_t    n_vocab                 = 0;
    uint32_t    wire_precision          = PIPE_SEGMENT_WIRE_F32;
    uint64_t    capabilities            = 0;
    uint64_t    cache_epoch             = 0;
    // What the head REQUIRES the terminal segment to return. A tail that cannot
    // serve this kind must reject the HELLO, never quietly send the other one.
    uint32_t    terminal_kind           = PIPE_SEGMENT_TERMINAL_HIDDEN;
    // INTERIOR TAPS the head REQUIRES this segment to extract, ascending, all inside
    // the segment's band. A segment that was not configured to extract exactly these
    // must reject the HELLO: silently returning none leaves the head conditioning a
    // speculative draft on a stale buffer, which does not change the verified output
    // and therefore cannot be caught by any parity test.
    std::vector<uint32_t> tap_layers;
    // Whether the head will actually READ the terminal segment's nextn sideband. True
    // only under --spec-type draft-mtp, whose drafter verifies against the target's
    // pre-LM-head hidden state; draft-dspark conditions on interior taps plus its own
    // draft context's nextn, and a no-spec run reads nothing. A tail must serialize the
    // sideband ONLY when this is set -- shipping it unasked cost a full duplicate
    // n_embd f32 run per token on the production arm.
    //
    // Declared as a need rather than inferred from terminal_kind because the two are
    // independent: the LOGITS arm also has a nextn sideband, and it is the only arm
    // where the sideband is genuinely different data from the terminal payload.
    uint32_t              nextn_need = 0;
};

struct pipe_segment_hello_ack {
    uint32_t    version = PIPE_SEGMENT_HELLO_VERSION;
    bool        accepted = false;
    uint32_t    n_vocab = 0;
    uint32_t    rs_snapshots = 0;
    // What this segment WILL return. On an accepted HELLO the head asserts this
    // equals what it asked for; terminal_width is the f32 column count that
    // implies (n_vocab for LOGITS, n_embd for HIDDEN) and is what the head sizes
    // its decode against. Non-terminal segments echo the request and report
    // terminal_width = n_embd.
    uint32_t    terminal_kind = PIPE_SEGMENT_TERMINAL_HIDDEN;
    uint32_t    terminal_width = 0;
    std::string reason;
    // Interior taps this segment WILL extract, echoing the request, plus the f32 column
    // count of one tap row (n_embd). tap_width is negotiated rather than assumed --
    // nextn_width is hardcoded on both sides today and only fails safe by accident.
    //
    // Declared AFTER `reason` on purpose, even though the wire order puts them before
    // it: this struct is aggregate-initialized positionally at several call sites, and
    // appending keeps every existing 7-field initializer valid.
    std::vector<uint32_t> tap_layers;
    uint32_t    tap_width = 0;
    // The f32 column count this segment WILL put in pipe_segment_fwd_resp::nextn, or 0
    // when it will send none. A terminal segment answers n_embd_out if the head declared
    // nextn_need and 0 otherwise; a non-terminal segment always answers 0. The head
    // asserts this against what it asked for, so a worker that ignores the need flag
    // fails at the handshake instead of appending bytes the head's frame-length check
    // would reject on the first forward.
    //
    // Appended for the same positional-initializer reason as tap_layers/tap_width above,
    // even though the wire puts it in the fixed region next to tap_width.
    uint32_t    nextn_width = 0;
};

// F32 activations are [n_tokens, n_embd] in token-major order. The header's
// seq_id and this payload seq_id must agree; carrying it in both places makes
// a bad or stale forwarding implementation detectable before it computes.
struct pipe_segment_fwd_req {
    uint32_t             version          = PIPE_SEGMENT_FWD_VERSION;
    uint64_t             session_id       = 0;
    uint64_t             seq_id           = 0;
    uint32_t             n_tokens         = 0;
    uint32_t             n_pos_per_token  = 1;
    uint32_t             n_seqs           = 1;
    std::vector<int32_t> positions;
    std::vector<uint32_t> seq_token_counts;
    std::vector<float>   activations;
};

struct pipe_segment_fwd_resp {
    uint32_t           version    = PIPE_SEGMENT_FWD_VERSION;
    uint64_t           session_id = 0;
    uint64_t           seq_id     = 0;
    uint32_t           n_tokens   = 0;
    uint32_t           output_width = 0;
    uint32_t           nextn_width = 0;
    // Interior taps: n_taps rows-blocks of [n_tokens, tap_width] f32, concatenated in
    // the ascending tap_layers order negotiated at HELLO. n_taps is carried explicitly
    // so the decoder can check the exact length without a second out-of-band value.
    uint32_t           tap_width = 0;
    uint32_t           n_taps = 0;
    std::vector<float> activations;
    std::vector<float> nextn;
    std::vector<float> taps;
    // NEXTN DEDUP. When set, `nextn` is bit-identical to `activations` and its f32 run is
    // OMITTED from the payload entirely -- the frame carries one copy, not two. Requires
    // nextn_width == output_width and nextn_width != 0.
    //
    // Decided PER RESPONSE by the tail, by comparing the two buffers it already holds,
    // NOT by assuming they must match. They coincide only under the HIDDEN terminal
    // payload and only for architectures whose t_h_nextn is the post-output_norm tensor
    // that t_embd is also taken from (qwen35.cpp:350-361 is the production case). Under
    // LOGITS they are categorically different (n_vocab logits vs n_embd hidden), and
    // architectures such as deepseek4.cpp:373 / dflash.cpp:418 set t_h_nextn to a
    // confidence vector instead. An equality TEST covers all of them without the
    // protocol having to know which model is loaded, and the reconstruction is exact by
    // construction, so nothing here can perturb a token.
    //
    // The DECODER leaves `nextn` empty and only reports the flag; reconstruction happens
    // in pipe-dense-segment-client, so encode/decode stays a pure byte transform and a
    // round-trip test still compares like with like.
    uint32_t           nextn_aliased = 0;
};

// RESET starts an empty cache epoch. KV_TRIM makes speculative rollback
// ordered at every segment. PROMPT_REUSE asks whether a content-addressed
// prefix of n_past tokens is available in this epoch.
struct pipe_segment_ctrl {
    uint32_t               version                = PIPE_SEGMENT_CTRL_VERSION;
    uint32_t               control                = PIPE_SEGMENT_CTRL_RESET;
    uint64_t               session_id             = 0;
    uint64_t               cache_epoch            = 0;
    uint32_t               n_past                 = 0;
    std::string            prompt_identity_sha256;
};

struct pipe_segment_ctrl_ack {
    uint32_t version     = PIPE_SEGMENT_CTRL_ACK_VERSION;
    uint32_t control     = PIPE_SEGMENT_CTRL_RESET;
    uint64_t session_id  = 0;
    uint64_t cache_epoch = 0;
    uint32_t status      = PIPE_SEGMENT_CTRL_APPLIED;
    uint32_t n_past      = 0;
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
std::vector<uint8_t> pipe_encode_expert_dispatch_begin(const pipe_expert_dispatch_begin & p);
std::vector<uint8_t> pipe_encode_expert_dispatch_acts(const pipe_expert_dispatch_acts & p);
std::vector<uint8_t> pipe_encode_expert_prefetch_hint(const pipe_expert_prefetch_hint & p);
std::vector<uint8_t> pipe_encode_expert_partial(const pipe_expert_partial & p);
std::vector<uint8_t> pipe_encode_segment_hello(const pipe_segment_hello & p);
std::vector<uint8_t> pipe_encode_segment_hello_ack(const pipe_segment_hello_ack & p);
std::vector<uint8_t> pipe_encode_segment_fwd_req(const pipe_segment_fwd_req & p, int32_t n_embd);
std::vector<uint8_t> pipe_encode_segment_fwd_resp(const pipe_segment_fwd_resp & p);
std::vector<uint8_t> pipe_encode_segment_ctrl(const pipe_segment_ctrl & p);
std::vector<uint8_t> pipe_encode_segment_ctrl_ack(const pipe_segment_ctrl_ack & p);
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
pipe_expert_dispatch_begin pipe_decode_expert_dispatch_begin(
    const uint8_t * buf, size_t len);
pipe_expert_dispatch_acts pipe_decode_expert_dispatch_acts(
    const uint8_t * buf, size_t len, uint32_t n_tokens, int32_t n_embd);
pipe_expert_prefetch_hint pipe_decode_expert_prefetch_hint(
    const uint8_t * buf, size_t len);
pipe_expert_partial pipe_decode_expert_partial(
    const uint8_t * buf, size_t len, int32_t n_embd);
pipe_segment_hello pipe_decode_segment_hello(const uint8_t * buf, size_t len);
pipe_segment_hello_ack pipe_decode_segment_hello_ack(const uint8_t * buf, size_t len);
pipe_segment_fwd_req pipe_decode_segment_fwd_req(const uint8_t * buf, size_t len, int32_t n_embd);
// tap_width / n_taps are what the head negotiated with THIS segment at HELLO; the
// decoder rejects a response that does not match, so a segment that silently stopped
// extracting fails here rather than leaving the head on a stale buffer.
//
// Deliberately NOT defaulted to 0. Defaults let an existing call site keep compiling
// while silently claiming "no taps negotiated", which turns a caller's oversight into a
// runtime frame mismatch instead of a compile error -- exactly how the tap-enabled
// worker roundtrip first failed. Every caller must state what it negotiated.
pipe_segment_fwd_resp pipe_decode_segment_fwd_resp(const uint8_t * buf, size_t len,
                                                    int32_t output_width, int32_t nextn_width,
                                                    int32_t tap_width, int32_t n_taps);
pipe_segment_ctrl pipe_decode_segment_ctrl(const uint8_t * buf, size_t len);
pipe_segment_ctrl_ack pipe_decode_segment_ctrl_ack(const uint8_t * buf, size_t len);

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
