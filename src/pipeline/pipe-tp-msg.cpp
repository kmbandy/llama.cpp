#include "pipe-tp-msg.h"

#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------------------------
// little helpers
// ---------------------------------------------------------------------------------------------

uint64_t pipe_tp_fnv1a(const void * data, size_t size, uint64_t seed) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = seed;
    for (size_t i = 0; i < size; i++) {
        h ^= (uint64_t) p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

uint64_t pipe_tp_model_hash(const char * desc, uint64_t n_params, uint32_t n_vocab, uint32_t n_layer) {
    uint64_t h = pipe_tp_fnv1a(desc ? desc : "", desc ? strlen(desc) : 0);
    h = pipe_tp_fnv1a(&n_params, sizeof(n_params), h);
    h = pipe_tp_fnv1a(&n_vocab,  sizeof(n_vocab),  h);
    h = pipe_tp_fnv1a(&n_layer,  sizeof(n_layer),  h);
    return h;
}

namespace {

template <typename T> void put(std::vector<uint8_t> & out, const T & v) {
    const size_t off = out.size();
    out.resize(off + sizeof(T));
    memcpy(out.data() + off, &v, sizeof(T));
}

template <typename T> void put_n(std::vector<uint8_t> & out, const T * v, size_t n) {
    if (n == 0) {
        return;
    }
    const size_t off = out.size();
    out.resize(off + n * sizeof(T));
    memcpy(out.data() + off, v, n * sizeof(T));
}

struct reader {
    const uint8_t * p;
    size_t          left;

    template <typename T> bool get(T * v) {
        if (left < sizeof(T)) {
            return false;
        }
        memcpy(v, p, sizeof(T));
        p    += sizeof(T);
        left -= sizeof(T);
        return true;
    }

    template <typename T> bool get_n(T * v, size_t n) {
        const size_t bytes = n * sizeof(T);
        if (n != 0 && (bytes / sizeof(T) != n || left < bytes)) {
            return false;
        }
        memcpy(v, p, bytes);
        p    += bytes;
        left -= bytes;
        return true;
    }
};

// Every frame ends with an FNV-1a over everything before it. A descriptor that has been truncated,
// reordered or partially overwritten then fails here instead of decoding into a plausible-looking
// batch that quietly desynchronises the two ranks.
void seal(std::vector<uint8_t> & out) {
    put(out, pipe_tp_fnv1a(out.data(), out.size()));
}

bool unseal(const uint8_t * data, size_t size, size_t * body_size, std::string * err) {
    if (size < sizeof(uint64_t)) {
        if (err) *err = "frame shorter than its checksum";
        return false;
    }
    const size_t body = size - sizeof(uint64_t);
    uint64_t want = 0;
    memcpy(&want, data + body, sizeof(want));
    if (want != pipe_tp_fnv1a(data, body)) {
        if (err) *err = "frame checksum mismatch";
        return false;
    }
    *body_size = body;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// HELLO
// ---------------------------------------------------------------------------------------------

std::string pipe_tp_hello_mismatch(const pipe_tp_hello & mine, const pipe_tp_hello & theirs) {
    char buf[512];

#define TP_EQ(field, what)                                                              \
    if (mine.field != theirs.field) {                                                   \
        snprintf(buf, sizeof(buf), "%s differs: this rank %llu, peer %llu", what,        \
                 (unsigned long long) mine.field, (unsigned long long) theirs.field);   \
        return buf;                                                                     \
    }

    TP_EQ(version,     "TP protocol version")
    TP_EQ(model_hash,  "model identity (arch/params/vocab/layers)")
    TP_EQ(n_world,     "--tp-world")
    TP_EQ(split_hash,  "-ts / tensor_split")
    TP_EQ(n_ctx,       "-c / n_ctx")
    TP_EQ(n_ctx_seq,   "n_ctx_seq")
    TP_EQ(n_rs_seq,    "recurrent-state rollback depth (n_rs_seq)")
    TP_EQ(n_batch,     "-b / n_batch")
    TP_EQ(n_ubatch,    "-ub / n_ubatch")
    TP_EQ(n_seq_max,   "--parallel / n_seq_max")
    TP_EQ(type_k,      "-ctk / cache type K")
    TP_EQ(type_v,      "-ctv / cache type V")
    TP_EQ(n_vocab,     "vocab size")
    TP_EQ(n_embd,      "n_embd")
    TP_EQ(n_layer,     "n_layer")
    TP_EQ(flags,       "context flags (kv_unified/flash_attn/embeddings/causal_attn/swa_full)")
#undef TP_EQ

    // The two windows must TILE the world: [0, n_local_0) and [n_local_0, n_world). This is the
    // half of the check that must NOT be an equality - it is the one thing the ranks are allowed,
    // and required, to disagree about.
    if (mine.rank_first == theirs.rank_first) {
        snprintf(buf, sizeof(buf), "both ranks claim world device %u as their first device",
                 mine.rank_first);
        return buf;
    }
    const pipe_tp_hello & lo = mine.rank_first < theirs.rank_first ? mine   : theirs;
    const pipe_tp_hello & hi = mine.rank_first < theirs.rank_first ? theirs : mine;
    if (lo.rank_first != 0) {
        snprintf(buf, sizeof(buf), "no rank owns world device 0 (lowest rank_first is %u)", lo.rank_first);
        return buf;
    }
    if (lo.rank_first + lo.n_local != hi.rank_first) {
        snprintf(buf, sizeof(buf),
                 "rank windows do not tile the world: [0,%u) then [%u,%u)",
                 lo.n_local, hi.rank_first, hi.rank_first + hi.n_local);
        return buf;
    }
    if (hi.rank_first + hi.n_local != hi.n_world) {
        snprintf(buf, sizeof(buf),
                 "rank windows do not cover the world: last window ends at %u, --tp-world is %u",
                 hi.rank_first + hi.n_local, hi.n_world);
        return buf;
    }
    return std::string();
}

// ---------------------------------------------------------------------------------------------
// CTRL
// ---------------------------------------------------------------------------------------------

void pipe_tp_encode_ctrl(const pipe_tp_ctrl & ctrl, std::vector<uint8_t> & out) {
    out.clear();
    put(out, ctrl.op_seq);
    put(out, ctrl.op);
    put(out, ctrl.b0);
    put(out, (uint16_t) 0);
    put(out, ctrl.a);
    put(out, ctrl.b);
    put(out, ctrl.c);
    put(out, ctrl.d);
    seal(out);
}

bool pipe_tp_decode_ctrl(const uint8_t * data, size_t size, pipe_tp_ctrl * out, std::string * err) {
    size_t body = 0;
    if (!unseal(data, size, &body, err)) {
        return false;
    }
    reader r{data, body};
    uint16_t pad = 0;
    if (!r.get(&out->op_seq) || !r.get(&out->op) || !r.get(&out->b0) || !r.get(&pad) ||
        !r.get(&out->a) || !r.get(&out->b) || !r.get(&out->c) || !r.get(&out->d)) {
        if (err) *err = "truncated CTRL frame";
        return false;
    }
    if (pad != 0 || r.left != 0) {
        if (err) *err = "malformed CTRL frame";
        return false;
    }
    if (out->op == PIPE_TP_CTRL_NONE || out->op > PIPE_TP_CTRL_SHUTDOWN) {
        if (err) *err = "unknown CTRL op";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// DECODE
// ---------------------------------------------------------------------------------------------

enum {
    TP_BATCH_HAS_POS    = 1u << 0,
    TP_BATCH_HAS_SEQ    = 1u << 1,
    TP_BATCH_HAS_LOGITS = 1u << 2,
};

bool pipe_tp_encode_batch(const llama_batch & batch, bool is_encode, uint32_t op_seq,
                          std::vector<uint8_t> & out, std::string * err) {
    if (batch.n_tokens <= 0) {
        if (err) *err = "batch has no tokens";
        return false;
    }
    if (batch.embd != nullptr) {
        // Spec D.2/R7: the follower is a TOKEN path by construction. An embd batch would also need
        // n_pos_per_embd positions per token, which is the exact bug tools/pipeline/
        // pipeline-stage.cpp exists to work around. Refuse loudly rather than ship half of it.
        if (err) *err = "embd batches are not mirrored across ranks (token batches only)";
        return false;
    }
    if (batch.token == nullptr) {
        if (err) *err = "batch has neither token nor embd";
        return false;
    }

    const int32_t n_tokens = batch.n_tokens;

    uint8_t present = 0;
    if (batch.pos)      present |= TP_BATCH_HAS_POS;
    if (batch.n_seq_id) present |= TP_BATCH_HAS_SEQ;
    if (batch.logits)   present |= TP_BATCH_HAS_LOGITS;
    // n_seq_id and seq_id are one option, not two: llama_batch_allocr reads seq_id only when
    // n_seq_id is set, and every producer in the tree sets or clears them together.
    if ((batch.n_seq_id == nullptr) != (batch.seq_id == nullptr)) {
        if (err) *err = "batch has exactly one of n_seq_id / seq_id set";
        return false;
    }

    int32_t n_seq_total = 0;
    if (present & TP_BATCH_HAS_SEQ) {
        for (int32_t i = 0; i < n_tokens; i++) {
            if (batch.n_seq_id[i] < 0) {
                if (err) *err = "batch has a negative n_seq_id";
                return false;
            }
            n_seq_total += batch.n_seq_id[i];
        }
    }

    out.clear();
    out.reserve(24 + (size_t) n_tokens * 12 + (size_t) n_seq_total * 4);
    put(out, op_seq);
    put(out, (uint8_t) (is_encode ? 1 : 0));
    put(out, present);
    put(out, (uint16_t) 0);
    put(out, n_tokens);
    put(out, n_seq_total);

    put_n(out, batch.token, (size_t) n_tokens);
    if (present & TP_BATCH_HAS_POS) {
        put_n(out, batch.pos, (size_t) n_tokens);
    }
    if (present & TP_BATCH_HAS_SEQ) {
        put_n(out, batch.n_seq_id, (size_t) n_tokens);
        for (int32_t i = 0; i < n_tokens; i++) {
            put_n(out, batch.seq_id[i], (size_t) batch.n_seq_id[i]);
        }
    }
    if (present & TP_BATCH_HAS_LOGITS) {
        put_n(out, batch.logits, (size_t) n_tokens);
    }
    seal(out);
    return true;
}

bool pipe_tp_decode_batch(const uint8_t * data, size_t size, pipe_tp_batch_desc * out, std::string * err) {
    size_t body = 0;
    if (!unseal(data, size, &body, err)) {
        return false;
    }
    reader r{data, body};

    uint8_t  kind = 0, present = 0;
    uint16_t pad = 0;
    int32_t  n_tokens = 0, n_seq_total = 0;
    if (!r.get(&out->op_seq) || !r.get(&kind) || !r.get(&present) || !r.get(&pad) ||
        !r.get(&n_tokens) || !r.get(&n_seq_total)) {
        if (err) *err = "truncated DECODE header";
        return false;
    }
    if (pad != 0 || kind > 1 || present > 7 || n_tokens <= 0 || n_seq_total < 0) {
        if (err) *err = "malformed DECODE header";
        return false;
    }
    out->is_encode = kind == 1;

    out->token.resize((size_t) n_tokens);
    if (!r.get_n(out->token.data(), (size_t) n_tokens)) {
        if (err) *err = "truncated DECODE tokens";
        return false;
    }

    out->pos.clear();
    if (present & TP_BATCH_HAS_POS) {
        out->pos.resize((size_t) n_tokens);
        if (!r.get_n(out->pos.data(), (size_t) n_tokens)) {
            if (err) *err = "truncated DECODE positions";
            return false;
        }
    }

    out->n_seq_id.clear();
    out->seq_id.clear();
    if (present & TP_BATCH_HAS_SEQ) {
        out->n_seq_id.resize((size_t) n_tokens);
        if (!r.get_n(out->n_seq_id.data(), (size_t) n_tokens)) {
            if (err) *err = "truncated DECODE n_seq_id";
            return false;
        }
        int64_t sum = 0;
        for (int32_t i = 0; i < n_tokens; i++) {
            if (out->n_seq_id[i] < 0) {
                if (err) *err = "negative n_seq_id in DECODE";
                return false;
            }
            sum += out->n_seq_id[i];
        }
        if (sum != n_seq_total) {
            if (err) *err = "DECODE n_seq_id does not sum to the declared total";
            return false;
        }
        out->seq_id.resize((size_t) n_seq_total);
        if (!r.get_n(out->seq_id.data(), (size_t) n_seq_total)) {
            if (err) *err = "truncated DECODE seq_id";
            return false;
        }
    }

    out->logits.clear();
    if (present & TP_BATCH_HAS_LOGITS) {
        out->logits.resize((size_t) n_tokens);
        if (!r.get_n(out->logits.data(), (size_t) n_tokens)) {
            if (err) *err = "truncated DECODE logits";
            return false;
        }
    }

    if (r.left != 0) {
        if (err) *err = "trailing bytes in DECODE frame";
        return false;
    }
    return true;
}

llama_batch pipe_tp_batch_desc::view() {
    llama_batch b = {};
    b.n_tokens = (int32_t) token.size();
    b.token    = token.data();
    b.embd     = nullptr;
    b.pos      = pos.empty() ? nullptr : pos.data();
    b.logits   = logits.empty() ? nullptr : logits.data();

    if (n_seq_id.empty()) {
        b.n_seq_id = nullptr;
        b.seq_id   = nullptr;
    } else {
        // llama_batch::seq_id is an array of pointers into a flat store; rebuild it here so the
        // follower hands llama_decode exactly the shape the leader had.
        seq_id_ptrs.assign(n_seq_id.size(), nullptr);
        size_t off = 0;
        for (size_t i = 0; i < n_seq_id.size(); i++) {
            seq_id_ptrs[i] = seq_id.data() + off;
            off += (size_t) n_seq_id[i];
        }
        b.n_seq_id = n_seq_id.data();
        b.seq_id   = seq_id_ptrs.data();
    }
    return b;
}
