#pragma once

#include "pipe-channel.h"
#include "pipe-dense-segment-manifest.h"
#include "pipe-protocol.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

struct common_params;

namespace wp_segment_worker {

struct options {
    std::filesystem::path manifest_path;
    uint32_t              segment_id = 0;
    uint32_t              recurrent_snapshots = 16;
};

struct resolved_segment {
    pipe_dense_segment::manifest manifest;
    pipe_dense_segment::segment  segment;
    std::filesystem::path        stage_gguf;
};

resolved_segment resolve_segment(const options & options);

class runtime {
public:
    virtual ~runtime() = default;

    // nextn_width is the width NEGOTIATED WITH THIS CONNECTION, not the segment's
    // capability: 0 means the head declared no need and the implementation must leave
    // response.nextn empty. It is an explicit parameter rather than a member because it
    // varies per connection while the runtime outlives none of them -- and because a
    // silently-ignored need is undetectable downstream (it only degrades a draft's
    // acceptance rate, never a verified token).
    virtual bool forward(const pipe_segment_fwd_req & request,
                         pipe_segment_fwd_resp & response,
                         uint32_t nextn_width,
                         std::string & error) = 0;
    virtual void reset() = 0;
    virtual bool trim(uint32_t n_past, std::string & error) = 0;
    virtual uint32_t n_past() const = 0;
};

struct service_config {
    uint32_t                    segment_id = 0;
    int32_t                     layer_first = -1;
    int32_t                     layer_last = -1;
    std::string                 model_identity_sha256;
    int32_t                     n_embd = 0;
    uint32_t                    n_vocab = 0;
    uint32_t                    output_width = 0;
    // The nextn sideband width this segment CAN produce (n_embd_out on a tail, 0
    // elsewhere). What it actually sends is min'd against the head's declared
    // nextn_need at HELLO -- see serve_connection(). Kept as a capability rather than a
    // decision because the graph is armed at load time
    // (llama_set_embeddings_nextn) and cannot be renegotiated, while serializing the
    // result is free to be conditional.
    uint32_t                    nextn_width = 0;
    uint32_t                    recurrent_snapshots = 0;
    uint64_t                    capabilities = 0;
    // What this segment returns in a terminal forward response. Decided at LOAD
    // time, not per connection: under PIPE_SEGMENT_TERMINAL_HIDDEN the tail sets
    // llama_set_no_output_head() so the LM head is never built into the graph, and
    // that cannot be renegotiated once the graph is reserved. A head asking for the
    // other kind is therefore rejected at HELLO. Non-terminal segments carry
    // PIPE_SEGMENT_TERMINAL_HIDDEN and it is not meaningful for them.
    uint32_t                    terminal_kind = PIPE_SEGMENT_TERMINAL_HIDDEN;
    // Only a terminal segment constrains terminal_kind. A middle segment returns
    // hidden state either way, so it must ACCEPT whatever the head negotiated --
    // otherwise flipping WP_SEGMENT_TAIL_LOGITS for an A/B would wrongly break
    // every middle worker in the chain.
    bool                        is_terminal = false;
    // INTERIOR TAPS this segment extracts, ascending, all inside [layer_first,
    // layer_last]. Taken from the manifest, which the head parses too, and armed at
    // LOAD time with llama_set_embeddings_layer_inp() -- that flag changes the graph
    // (it marks t_layer_inp[il] a graph output) and cannot be renegotiated once the
    // graph is reserved, exactly like terminal_kind. A head asking for a different set
    // is therefore rejected at HELLO rather than quietly served nothing: the head would
    // then read a stale buffer, condition its draft on it, and still emit bit-identical
    // verified output, so nothing downstream could detect the fault.
    std::vector<uint32_t>       tap_layers;
};

// Serve one client connection. Requests are handled serially, so a control
// acknowledgement is always sent before a later forward request is read.
int serve_connection(pipe_channel::channel & channel, const service_config & config, runtime & runtime);
int run(const options & options, common_params & params);

} // namespace wp_segment_worker
