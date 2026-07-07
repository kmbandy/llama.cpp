# MoE-Aware Batched Paging — Design Spec

**Date:** 2026-07-07
**Branch:** `feat/wp-vnext`
**Status:** Design approved; implementation plan pending (writing-plans next)

## Goal

Let the weight-pager's eval-callback batching engage on **MoE models under real paging
pressure** (partial residency + evictions), so large MoE decode (300–700B class, e.g.
DeepSeek V4 Flash 284B-A13B) removes the per-op GPU-sync tax and opens compute windows
for expert page-in I/O — without the near-null GPU fault that batching currently causes on
any MoE.

## Background: why batching faults on MoE today (verified root cause)

`WP_BATCH_EVAL_CB` (shipped, default-on for dense) makes the eval-callback return `false`
so the ggml scheduler batches many ops into one async compute instead of syncing after each
node (ggml/src/ggml-backend.cpp:1707-1736). On dense-resident models this gives native decode
speed + native PPL. On **every** MoE it hard-faults (`Memory access fault … Page not present`,
near-null address) in the routed `mul_mat_id` (MMQ) kernel. Reproduced on a clean valid-PPL
MoE (LFM2.5-8B-A1B): per-op paged PPL 27.09 == native 27.23; batching → hard fault.

Two independent, verified failure modes — both are the same root cause (batching breaks the
"one MoE op fully processed at a time" invariant the design relies on):

- **H3 (read-before-produce), primary, source-verified.** The eval-cb for a `MUL_MAT_ID`
  reads the router's selected-expert ids (`idx_tensor = t->src[2]`, wp-eval-cb.cpp:431) via a
  D2H copy + `hipStreamSynchronize` (wp-eval-cb.cpp:502-520) to build the active-expert set.
  In per-op mode the router kernel that *produces* those ids has already computed+synced. In
  batched mode the router is a **later, unlaunched node in the pending range**, so the sync
  waits on a stream that doesn't yet contain the router kernel and returns immediately over a
  stale buffer → garbage ids → all filtered out (`if (idx<0||idx>=n_subs) continue;`) → empty
  active set → `host_ptrs` all-null → sentinel-fill skipped (first_active_slot null) →
  `s_dev_expert_ptrs` memcpy'd all-null → MMQ kernel indexes a null base → near-null fault.
- **H4 (TLS take-steal), secondary, real.** `ggml_cuda_take_routed_expert_ptrs()` (mmq.cu:52)
  reads-and-clears the routed-expert TLS at MMQ **launch**. It is called by
  `ggml_cuda_mul_mat_q` (mmq.cu:265), which serves **both** `MUL_MAT_ID` (ggml-cuda.cu:2959)
  **and regular** quantized `mul_mat` (ggml-cuda.cu:2889, ids=nullptr). So an ordinary MMQ op
  batched ahead of the routing op consumes the TLS meant for it, leaving the real routing op
  with null → placeholder deref → fault.

Refuted hypotheses (recorded so we don't revisit): **H2 cross-stream race** — refuted, the
expert-ptr `hipMemcpyAsync` and the MMQ kernel run on the **same** stream
(`ggml_cuda_set_wp_compute_stream(cuda_ctx->stream())`, ggml-cuda.cu:4991; returned verbatim by
mmq.cu:120), so same-stream FIFO already orders them. **H1 pin/TLS clobber during
range-building** — refuted under `evictions==0` (unpinned slots aren't recycled) and because a
routing op is the range end so nothing runs between its set and the compute.

**Isolating each `MUL_MAT_ID` into its own singleton range fixes both:** the prior range
(incl. router) computes+syncs first → ids valid (kills H3); no other op shares the range to
`take()`-steal (kills H4). The shipped dense-only guard (`batch_safe()` requires
`!catalog_.has_experts()`, wp-pager.cpp:506) is a correct-but-coarse version of this: it
isolates *every* op on a MoE model. This spec replaces that coarseness with per-boundary
isolation so the dense sub-ranges can batch.

## Target regime (explicit)

**Production is partial residency, not fully resident.** The models this exists for
(300–700B MoE) cannot hold all experts in VRAM; prefetch aims to make *most* active experts
resident in time, but there are always misses and — critically — **evictions every token**.
Consequences that shape the design:

1. The ids read is **essential** (must know which experts to page in *and* build the
   pointer array from only-active, only-resident experts). Not skippable.
2. Today `batch_safe()` requires `evictions==0`, which is true only fully-resident. **So
   batching cannot engage at all in production today.** Removing that requirement — safely —
   is the core of this work.

## Architecture

Five units, each independently testable.

### A. Routing-boundary pre-pass (`WeightPager::mark_routing_boundaries`)

- **What:** one walk over the full `ggml_cgraph` before the scheduler splits it. For every
  `GGML_OP_MUL_MAT_ID` node `M`: (1) add `M` to an isolate set; (2) resolve `M->src[2]`
  through its `view_src` chain to the producing root tensor `R` and add `R` to a
  break-after set. Store the union as `std::unordered_set<const ggml_tensor*>
  routing_break_tensors_` on the pager.
- **Where called:** `llama_context::graph_compute` (src/llama-context.cpp:2473-2492), before
  `ggml_backend_sched_graph_compute_async`. The pager handle is already available
  (`model.wp_pager`, used at the eval-cb registration, llama-context.cpp:1365-1366).
- **Why tensor-identity, not node-index:** the scheduler re-orders/splits nodes, but split
  nodes are *the same `ggml_tensor` objects*. Marking by pointer survives splitting; "the node
  before it in index order" does not.
- **Caching:** recompute only when graph topology changes. Signature = `(n_nodes, nodes[0],
  nodes[n_nodes-1])` (pointer identity + count); store last signature, skip the walk on match.
  Decode graphs are stable across tokens, so this amortizes to ~0.
- **Interface:**
  - `void WeightPager::mark_routing_boundaries(const struct ggml_cgraph * gf);`
  - `bool WeightPager::is_routing_break(const struct ggml_tensor * t) const;` (O(1) set lookup)

### B. Router-boundary handling (reuse the validated per-op MoE path)

At an isolated `MUL_MAT_ID`, the prior dense range (including the router and the ids-producer
`R`) has computed **and synced** (ggml-backend.cpp:1729), so `src[2]->data` is materialized.
The callback then runs **exactly today's per-op MoE code** — read ids, determine active
experts, `ensure()` (page in misses) + pin, build `s_dev_expert_ptrs`, set TLS, compute.
Isolation guarantees no other op is in the range to `take()`-steal. **No change to this code
path** beyond arranging for it to run isolated.

### C. Batched pin lifecycle with reactive auto-break (the core new work)

Today the pin drain unpins the *previous op's* slots at the top of every callback
(wp-eval-cb.cpp:337) — safe only because `evictions==0` means an unpinned slot is never
recycled. The change:

- **Per-range accumulation:** pin every page an op in the current range touches into
  `s_current_range_pins`; do **not** unpin at each callback.
- **Release after compute+sync:** a range's pins are released only once its compute+sync has
  run. Release points (belt-and-suspenders, since not every range ends with an `ask=false`):
  - primary: the `ask=false` callback on the range's last node (ggml-backend.cpp:1731) — fires
    after the sync at line 1729;
  - fallback: at the top of the first `ask=true` callback of the next range, if a
    boundary-release is pending;
  - teardown: `weight_pager_eval_cb_reset` / graph end flushes any residual range pins.
- **Reactive auto-break on pin exhaustion:** if pinning a page during range-building cannot be
  satisfied without evicting a slot still pinned by an earlier op in the current range, the
  callback returns `true` **immediately** — ending the range, forcing compute+sync, releasing
  its pins — then the next range continues. Ranges therefore grow to the size free VRAM allows
  and shrink automatically under eviction pressure. No static budget, no tuning knob.
- **Result:** `batch_safe()` drops the `evictions==0` and `!has_experts()` requirements
  entirely; batching is governed by *live pinnability* plus the routing-boundary breaks.

### D. I/O overlap + explicit scope boundary

- Expert `ensure()` uses the existing async path (`WP_ASYNC_ENSURE`) so page-in I/O overlaps
  compute; the batched dense ranges provide the compute windows.
- **Out of scope (separate lever):** *predictive/speculative* prefetch of a future layer's
  experts before its router runs. This design overlaps only I/O it can issue once ids are known.

### E. Flag gating

All new behavior behind a single default-off flag, `WP_PAGED_BATCH` (name provisional). When
**off**: byte-identical to today (dense resident-batching via `WP_BATCH_EVAL_CB`; MoE and any
eviction → per-op). When **on**: pre-pass + boundary breaks + batched per-range pin lifecycle
engage, and `batch_safe` loses `evictions==0`/`!has_experts()`. Flip default-on only after both
validation gates pass.

## Interfaces summary

| Unit | Symbol | File |
|---|---|---|
| A | `WeightPager::mark_routing_boundaries(const ggml_cgraph*)` | wp-pager.{h,cpp} |
| A | `WeightPager::is_routing_break(const ggml_tensor*) const` | wp-pager.{h,cpp} |
| A | call site before sched compute | llama-context.cpp (~2492) |
| C | per-range pin accumulation + release | wp-eval-cb.cpp |
| C | reactive break on pin-exhaustion | wp-eval-cb.cpp (`eval_cb_op_return`) |
| C/E | `batch_safe()` loses `evictions==0`/`!has_experts` when flag on | wp-pager.cpp:506 |
| E | `WP_PAGED_BATCH` env gate | wp-eval-cb.cpp |

## Validation staging (flag-gated, default-off until both pass)

1. **Resident gate (`evictions==0`).** LFM2.5-8B-A1B (slots=750) and Qwen3.6-27B (slots=345).
   - Correctness: LFM PPL == **27.09**, 27B PPL == **5.4623**, both exact, **0 GPU faults**.
   - Perf: decode t/s ≥ per-op baseline (proves range-shaping + isolation are correct in
     isolation from the eviction machinery).
2. **Paged gate (`evictions>0`, the production target).** DeepSeek V4 Flash Q8 (~162GB) on the
   R9700 (32GB) — genuinely exceeds VRAM, so evictions happen every token. Shared/dense weights
   (embeddings, attention, indexer, router) stay hot; 13B active experts stream from NVMe.
   - Correctness: paged PPL matches native (within reduction-order noise), **0 GPU faults**,
     `routing_ptrs_discarded_unconsumed` sane.
   - Perf: decode t/s up vs per-op; measure prefetch hit-rate + I/O-overlap.
   - Note: V4 Flash's lightning-indexer / sparse-attention adds extra dense paged ops per
     layer — these are exactly what the batched dense sub-ranges accelerate, raising the payoff.

Validate via **llama-perplexity** (PPL) and **llama-server** `/completion` (decode t/s). Avoid
llama-cli (parks interactive). Heads-up: V4 Flash previously tripped a llama-server startup hang
in the `common_params_fit` dry-run under `--direct-io`; llama-perplexity should not hit it.

## Risks / open items (resolve during implementation)

- **`WP_ASYNC_ENSURE` composition:** the async path already has a deferred-unpin mechanism
  (`s_pending_async_ops`, wp-eval-cb.cpp:276-331). The per-range pin lifecycle must compose
  with it — likely the range-release must also drain/await the async completion events before
  unpinning. Treat as a first-class integration point in the plan.
- **view_src resolution for `src[2]`:** confirm the router ids reach `MUL_MAT_ID->src[2]`
  through views (reshape/cont) and that walking `->view_src` to the root yields the graph node
  the eval-cb is actually called on. Add a debug assert that every marked `R` is seen by the
  eval-cb at least once per token.
- **Multiple `MUL_MAT_ID` per layer** (gate/up + down): each returns `true`, so each is its own
  singleton range — the design already guarantees at most one routing op per range. Verify on
  V4 Flash (which also has shared-expert + routed-expert matmuls).
- **Cross-split boundaries:** if a `MUL_MAT_ID` and its ids-producer land in different backend
  splits, the split boundary already syncs (events) — confirm this doesn't double-break or
  strand pins.
- **Pin-exhaustion break correctness:** ensure the auto-break path leaves the TLS/expert state
  consistent (a break is just an earlier range end; the routing op still runs isolated later).
- **Concurrent multi-stream** (`curr_stream_no != 0`): a latent separate bug (the wp compute
  stream is captured once as stream 0); out of scope here but note it must stay disabled with
  paging.

## Out of scope

- Predictive/speculative expert prefetch (separate lever).
- The gpt-oss-20b NaN (gpt-oss/MXFP4-specific; benign `discarded_unconsumed:6` counter is not
  the cause; not on this critical path).
- Concurrent multi-stream scheduling with weight paging.
