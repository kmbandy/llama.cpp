#pragma once

// Cross-host tensor parallelism, milestone M3: the LOCKSTEP CONTROL MESSAGES.
//
// M2 gave the two ranks a shared connection and one message type on it (REDUCE, the per-reduce
// partial exchange). That is enough to make one llama_decode() call agree across two processes,
// but only if both processes call llama_decode() with the SAME batch, in the same order, against
// the same memory state. Nothing made that true: rank 1 had no way to be told what to decode.
//
// This file is that channel's payload format. It is deliberately free of ggml and of everything
// in src/ except include/llama.h's plain-old-data batch types, so the encoder and decoder can be
// compiled and round-tripped on a CPU with no model, no GPU and no socket (tests/test-tp-msg.cpp).
//
// WHY THE BATCH AND NOT THE UBATCH.  llama_context::decode() splits a batch into ubatches with a
// deterministic function of (batch, cparams.n_ubatch, memory state, n_seq_max). All three inputs
// are already equal across the ranks - the batch because it is mirrored here, the rest because
// HELLO refuses to start when they differ - so mirroring the BATCH makes the ubatch split agree
// for free, exactly as M1 derived the row split from world state rather than shipping it. A wide
// prefill therefore costs ONE descriptor, not one per ubatch, and MTP verify batches (which are
// ordinary token batches, spec D.2b) are mirrored with no extra message type.
//
// WHY TOKENS AND NOT HIDDEN STATES (spec D.2, R7).  The follower decodes token ids. That is what
// keeps the M-RoPE position hazard of tools/pipeline/pipeline-stage.cpp:18-46 - n_pos_per_embd
// == 4 for an IMROPE model while llama_batch_init allocates n_tokens positions - out of this
// design entirely: token batches broadcast one position across the sections. An embd batch is
// REJECTED by the encoder rather than half-supported.
//
// FRAMING.  Every payload here travels inside the M2 16-byte pipe_tp_frame_hdr (type byte +
// length), on the same connection as the REDUCE frames. Ordering does the rest: rank 0 sends a
// DECODE/CTRL and then enters the graph, so its reduce frames strictly follow the descriptor that
// explains them, and rank 1 has fully applied the descriptor before its first reduce.
//
// LITTLE-ENDIAN, like the M2 header, and for the same reason (both hosts are x86-64).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "llama.h"

// ---------------------------------------------------------------------------------------------
// HELLO - the startup equality gate (spec D.2, modelled on
// tools/wp-segment-worker/wp-segment-worker.cpp:25-37 hello_matches)
// ---------------------------------------------------------------------------------------------

#define PIPE_TP_PROTO_VERSION 1

// Every field except the two marked PER-RANK must be equal on both ranks or the run is refused at
// startup. A silent mismatch here is a wrong answer or a deadlock a hundred layers later.
struct pipe_tp_hello {
    uint32_t version;           // PIPE_TP_PROTO_VERSION
    uint32_t n_world;
    uint32_t rank_first;        // PER-RANK
    uint32_t n_local;           // PER-RANK
    uint32_t n_ctx;
    uint32_t n_ctx_seq;
    uint32_t n_rs_seq;          // recurrent-state snapshots per seq: the rollback depth
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t type_k;            // ggml_type of the K cache
    uint32_t type_v;
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_layer;
    uint32_t flags;             // see PIPE_TP_HELLO_F_*
    uint64_t model_hash;        // model identity: arch + params + vocab, see pipe_tp_model_hash()
    uint64_t split_hash;        // the tensor_split vector, as the world sees it
};

enum {
    PIPE_TP_HELLO_F_KV_UNIFIED  = 1u << 0,
    PIPE_TP_HELLO_F_FLASH_ATTN  = 1u << 1,
    PIPE_TP_HELLO_F_EMBEDDINGS  = 1u << 2,
    PIPE_TP_HELLO_F_CAUSAL_ATTN = 1u << 3,
    PIPE_TP_HELLO_F_SWA_FULL    = 1u << 4,
};

// Returns an empty string when the two are compatible, otherwise a human-readable description of
// the FIRST field that differs. `mine`/`theirs` are the two ranks' own HELLOs; rank_first and
// n_local are checked for consistency (the windows must tile [0, n_world) with no gap or overlap)
// rather than for equality.
std::string pipe_tp_hello_mismatch(const pipe_tp_hello & mine, const pipe_tp_hello & theirs);

// ---------------------------------------------------------------------------------------------
// CTRL - memory mutations
// ---------------------------------------------------------------------------------------------

enum pipe_tp_ctrl_op : uint8_t {
    PIPE_TP_CTRL_NONE      = 0,
    PIPE_TP_CTRL_MEM_CLEAR = 1, // llama_memory_clear(mem, b0)
    PIPE_TP_CTRL_SEQ_RM    = 2, // llama_memory_seq_rm  (mem, a, b, c)
    PIPE_TP_CTRL_SEQ_CP    = 3, // llama_memory_seq_cp  (mem, a, b, c, d)
    PIPE_TP_CTRL_SEQ_KEEP  = 4, // llama_memory_seq_keep(mem, a)
    PIPE_TP_CTRL_SEQ_ADD   = 5, // llama_memory_seq_add (mem, a, b, c, d)
    PIPE_TP_CTRL_SEQ_DIV   = 6, // llama_memory_seq_div (mem, a, b, c, d)
    PIPE_TP_CTRL_SHUTDOWN  = 7, // leader is going away; follower exits 0
};

struct pipe_tp_ctrl {
    uint32_t op_seq = 0;    // the shared lockstep counter, see below
    uint8_t  op     = PIPE_TP_CTRL_NONE;
    uint8_t  b0     = 0;    // bool argument (MEM_CLEAR's `data`)
    int32_t  a = 0, b = 0, c = 0, d = 0;
};

void pipe_tp_encode_ctrl(const pipe_tp_ctrl & ctrl, std::vector<uint8_t> & out);
bool pipe_tp_decode_ctrl(const uint8_t * data, size_t size, pipe_tp_ctrl * out, std::string * err);

// ---------------------------------------------------------------------------------------------
// DECODE - the per-step batch descriptor
// ---------------------------------------------------------------------------------------------

// THE LOCKSTEP COUNTER. Rank 0 increments one counter for EVERY mirrored operation, decode and
// control alike, and stamps it into the frame; rank 1 keeps its own and refuses anything that is
// not exactly one greater than the last. That single number catches a dropped mirror (an
// unaudited llama_memory_* call site), a reordering, and a duplicated send - the three ways this
// design can silently diverge - at the first frame after the mistake instead of a hundred layers
// later inside a reduce.

struct pipe_tp_batch_desc {
    uint32_t op_seq   = 0;
    bool     is_encode = false;

    std::vector<llama_token>  token;
    std::vector<llama_pos>    pos;       // empty => the batch had pos == nullptr
    std::vector<int32_t>      n_seq_id;  // empty => the batch had n_seq_id/seq_id == nullptr
    std::vector<llama_seq_id> seq_id;    // flattened, n_seq_id[i] entries per token
    std::vector<int8_t>       logits;    // empty => the batch had logits == nullptr

    // Pointers into the vectors above. Valid until this object is modified or destroyed.
    // Deliberately NOT a copy: llama_decode does not take ownership and does not outlive the call.
    llama_batch view();

private:
    std::vector<llama_seq_id *> seq_id_ptrs; // built by view()
};

// Returns false (and sets *err) for a batch this channel refuses to carry: an embd batch, or one
// with a negative/zero token count. `op_seq` is stamped into the frame.
bool pipe_tp_encode_batch(const llama_batch & batch, bool is_encode, uint32_t op_seq,
                          std::vector<uint8_t> & out, std::string * err);

bool pipe_tp_decode_batch(const uint8_t * data, size_t size, pipe_tp_batch_desc * out, std::string * err);

// ---------------------------------------------------------------------------------------------

// FNV-1a 64. Used for the frame checksums and for the HELLO identity hashes; not a security
// property, just a cheap way to make a truncated or reordered descriptor loud.
uint64_t pipe_tp_fnv1a(const void * data, size_t size, uint64_t seed = 0xcbf29ce484222325ull);

// Identity of the model as far as the row split is concerned.
uint64_t pipe_tp_model_hash(const char * desc, uint64_t n_params, uint32_t n_vocab, uint32_t n_layer);
