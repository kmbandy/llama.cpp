# Spec: cross-machine pipeline parallelism with per-machine weight paging

## Goal

Run GLM-5.2 (254 GB, UD-Q2_K_XL) split across mad-lab-main and mad-lab-2026, where **each
machine pages its own shard from its own NVMe**. main takes the first ~54 MoE layers, 2026 the
last ~21.

## Why RPC cannot do this (settled, do not re-litigate)

> ggml-rpc-server executes remote tensor ops on client-allocated tensors -- it has no model, no
> loader, no catalog, so there is nothing there to page. Paging on the 2026 side requires each
> machine running a FULL llama.cpp instance owning its own shard, joined at a layer boundary =
> pipeline parallelism, which llama.cpp does not have.

Also confirmed: `cparams.pipeline_parallel` (llama-context.cpp:397) is ggml_backend_sched's
INTRA-PROCESS multi-GPU batch splitting, and it is force-disabled under weight paging at
line 407. It is not related to this work and must not be confused with it.

## Set expectations honestly

For **single-stream decode this is not a throughput win**. The stages are strictly sequential --
main computes layers 0..56 while 2026 idles, then 2026 computes 57..77 while main idles -- so
token time is the SUM of both stages, not the max. The drives never overlap, because 2026 cannot
know which experts it needs until it receives the activation.

Measured/derived expectation: **~+10%**, entirely from added cache capacity (fleet pool 50 -> 70 GB,
residency 19% -> 27%), plus the ability to hold models too large for one box. Anyone reporting a
2x from this is measuring something else. The 2x requires MULTIPLE TOKENS IN FLIGHT, which is out
of scope here.

Build it for the capacity and as the foundation, not for the speed.

## Architecture

Two (later N) full llama.cpp processes, each owning a **contiguous layer band**:

```
HEAD (main)                          TAIL (2026)
  token_embd                           layers 57..77
  layers 0..56                         output_norm, output (lm_head)
  own KV for 0..56                     own KV for 57..77
  own pager + NVMe shard               own pager + NVMe shard
        |                                     ^
        |  hidden state [n_embd x n_tok] ---->|
        |<---------------- sampled token id --|
```

- Only the **activation** crosses: n_embd=6144. F32 = 24 KB/token, F16 = 12 KB/token.
- **Zero cross-machine expert traffic.** KV never crosses.
- Head owns embeddings; tail owns lm_head and sampling, and returns a **token id (4 bytes)**, not
  logits.

## Phase 1 -- layer-range model loading (IMPLEMENT THIS FULLY)

Let one process load and run only layers [first, last].

1. `llama_model_params` gains `int32_t pipeline_layer_first` / `pipeline_layer_last`
   (default -1 = own everything, which MUST reproduce today's behaviour exactly).
2. `load_tensors`: skip per-layer tensors outside the band. Load `token_embd` only when
   `first == 0`; load `output_norm` / `output` only when `last == n_layer-1`.
3. **`hparams.n_layer` stays GLOBAL.** RoPE, position handling and any absolute-layer logic must
   keep seeing the real depth. Introduce separate owned-range accessors; do NOT renumber layers.
4. The arch graph builder iterates only owned layers, taking its input from the supplied hidden
   state rather than from the embedding when `first != 0`.
5. KV cache allocates only for owned layers.
6. The weight pager catalogs only owned layers (it already keys on the tensors actually loaded,
   so this should follow, but assert it).

### Phase 1b -- the stage splitter

Each stage needs a loadable GGUF containing its own layers plus the shared tensors it owns.
Both machines loading the full 254 GB is impossible: **2026 has ~108 GB free**.

Write a tool (or extend `tools/wp-repack`, which already does layer-range sharding) that emits a
per-stage GGUF: the band's layer tensors, plus `token_embd` for the head and
`output_norm`/`output` for the tail, plus all KV metadata unchanged.

## Phase 2 -- the pipeline protocol (DESIGN CONCRETELY, implement if Phase 1 lands cleanly)

- Roles: HEAD / MIDDLE / TAIL.
- Request: `{seq_id, n_tokens, positions[], hidden[n_embd * n_tokens]}`.
- Response: hidden state (middle) or sampled token ids (tail).
- Length-prefixed over TCP. `ggml/src/ggml-rpc` has usable socket helpers -- reuse the transport
  plumbing, NOT the RPC protocol semantics.
- F16 on the wire halves it to 12 KB/token; state the precision assumption explicitly rather
  than assuming it is free.
- **Test it as a LOOPBACK ON ONE MACHINE first** -- both stages on main over localhost. That is
  fully verifiable without the second box and without a network variable.

## Phase 3 -- cross-machine (DESIGN ONLY for now)

Same protocol over Tailscale (main 100.86.191.92, 2026 100.102.191.30). Measured RTT is
0.5-0.6 ms, so ~1 ms/token of crossing against a ~2000 ms token. Negligible; do not optimise it.

## Invariants

- **`pipeline_layer_first/last` unset must be byte-identical to today.** This is the regression
  that matters most; everything else is new code.
- A stage must **refuse to start** if its band is empty, discontinuous, or if head/tail roles are
  inconsistent (e.g. nobody owns `token_embd`). Fail loudly at load, never silently produce a
  model that runs and emits garbage.
- Absolute layer indices are preserved end to end. A tensor named `blk.57.*` is layer 57 on every
  machine.
- The existing weight-paging invariants still hold: `is_paged_weight()`, the catalog `page_this`
  filter and the buft override must continue to agree (see comments at those sites).

## Constraints

- Do NOT run any GPU work, any model, any inference, `llama-cli`/`llama-server`, or cmake/make/
  ninja builds. `g++ -fsyntax-only` and standalone CPU unit tests are fine.
- Do NOT run `npx gitnexus analyze` or any gitnexus tooling; the repo CLAUDE.md tells you to,
  ignore it (it consumed 5.9 GB last time).
- Do NOT commit, stash, revert, or `git checkout` anything. **The tree has ~25 uncommitted files**
  from today (multi-device paging, resident-expert blocks, Vulkan prefetch, server metrics).
  Build on top; do not touch them.
- ASCII only in code and comments -- no Unicode arrows or em-dashes.
- A HIP build may be running concurrently. Do not start another.

## Deliverable

Implement **Phase 1 + 1b** completely with unit tests, and produce a concrete written design for
Phase 2. Prefer a smaller correct result with honest gaps to a large unverified one. Report which
files changed, what you tested with actual output, and anything you could not satisfy.
