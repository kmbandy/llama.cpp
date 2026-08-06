#pragma once

// Hash-layer expert oracle: token id -> expert ids, exactly, with no model
// forward and no prediction.
//
// DS4-Flash's first `dsv4_hash_layer_count` blocks (3 on this checkpoint, layers
// 0..2) do not route with a learned gate. deepseek4.cpp:1809 selects their
// experts with
//
//     selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens)
//
// -- a pure table lookup on the token id. This class is the CPU-side copy of
// exactly that lookup, so for those layers "which experts will this token need"
// is answerable the moment the token id is known, which for prefill is before
// the first read and for speculative decode is a whole draft block ahead.
//
// WHY THIS EXISTS SEPARATELY FROM WeightPager::collect_tid2eid_pages_():
// that function does the same lookup and then resolves expert -> PAGE against
// the pager catalog. On the cross-machine layout the second half cannot work and
// the first half is unreachable:
//   - the spine runs the dense model without --weight-paging, so
//     model.wp_pager is null and llama_wp_on_draft_tokens returns 0 at its null
//     check (llama-model.cpp:4070) before reading any table;
//   - even with a pager, deepseek4.cpp:261 marks the routed experts
//     TENSOR_SKIP | TENSOR_NOT_REQUIRED under cross-machine dispatch, so the
//     spine's catalog holds no routed-expert pages to resolve to.
// The expert -> page half belongs on the worker, which owns the shard catalog.
// So the lookup is lifted out here, free of the pager, and stops at expert ids
// -- the last representation both machines can agree on.
//
// Deliberately free of ggml and llama-model: the table is handed in as plain
// host memory by whoever loads the tensor. That keeps this unit-testable with no
// model, and keeps a wrong table shape a load-time error rather than a silent
// mis-index at dispatch time.
//
// THREADING: register_layer() is load-time only. After the last registration the
// object is immutable and experts_for() is const and reentrant -- it allocates
// its own scratch and touches no shared state. Do not interleave the two.

#include <cstdint>
#include <vector>

namespace pipe_expert_dispatcher {

class hash_oracle {
  public:
    // Register the host copy of blk.<layer>.ffn_gate_tid2eid.
    //
    // `data` is n_vocab rows of n_expert_used int32 expert ids, row-major by
    // token id -- the tensor is created as {n_expert_used, n_vocab}, so ne[0] is
    // the row stride. Copied; the caller keeps ownership.
    //
    // n_expert is the model's expert count and bounds what the table may
    // contain. An id outside [0, n_expert) means the table and the model
    // disagree, which throws rather than being clamped: a silently clamped id
    // would send a worker to prefetch the wrong expert, and the resulting
    // "prefetch does not help" measurement would be indistinguishable from the
    // real thing.
    //
    // Throws std::invalid_argument on a bad shape or an out-of-range id, and on
    // a duplicate layer.
    void register_layer(int32_t         layer,
                        int32_t         n_expert_used,
                        int32_t         n_vocab,
                        int32_t         n_expert,
                        const int32_t * data);

    bool empty() const { return tables_.empty(); }

    // Drop every table. The caller that registers a set of layers is the only
    // one that knows the set is a UNIT: half a hash block registered is worse
    // than none, because it would hint some layers and silently not others,
    // and the resulting half-measurement would look like a weak positive.
    void clear() {
        tables_.clear();
        layers_.clear();
    }

    // Layers that have a table, ascending. These are exactly the layers
    // experts_for() can answer for.
    const std::vector<int32_t> & layers() const { return layers_; }

    // The union of experts `tokens` select on `layer`, written to `out`
    // ASCENDING and DEDUPED -- the order and shape pipe_expert_prefetch_hint
    // requires on the wire. `out` is cleared first.
    //
    // Returns false (leaving `out` empty) when the layer has no table, which is
    // the normal answer for every layer past the hash block. Token ids outside
    // [0, n_vocab) are skipped, not an error: a draft model can propose an id
    // this table does not cover, and a speculative hint has no business
    // throwing.
    //
    // Negative entries in the table are skipped -- the tensor uses them as
    // "unused slot" padding when a row selects fewer than n_expert_used.
    bool experts_for(int32_t                layer,
                     const int32_t *        tokens,
                     size_t                 n_tokens,
                     std::vector<int32_t> & out) const;

  private:
    struct table {
        int32_t              layer         = -1;
        int32_t              n_expert_used = 0;
        int32_t              n_vocab       = 0;
        int32_t              n_expert      = 0;
        std::vector<int32_t> data;
    };

    const table * find(int32_t layer) const;

    std::vector<table>   tables_;
    std::vector<int32_t> layers_;
};

}  // namespace pipe_expert_dispatcher
