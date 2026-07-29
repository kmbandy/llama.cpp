# Spec: Phase 2 pipeline protocol — loopback on mad-lab-main

Repo: `/home/kmbandy/GitHub/llama.cpp`, branch `master`. Build on top of the working
tree; other agents have work in it.

**Read first, in this order. They are the contract, not background:**
1. `docs/dev/2026-07-28-pipeline-protocol-design.md` — the full Phase 2 design:
   wire format, frame types, roles, transport decision, sampler ownership,
   failure model, and the loopback test plan. **Implement that document.** Do not
   redesign the protocol. If you believe something in it is wrong, say so in your
   report and stop — do not silently deviate.
2. `src/llama-pipeline.h` — Phase 1, already implemented and unit-tested.
   `llama_pipeline_owns_tensor` and `llama_pipeline_validate_stages` are the
   single source of truth for stage membership and pipeline consistency. Call
   them; do not reimplement the predicates.
3. `docs/superpowers/specs/2026-07-28-cross-machine-pipeline-parallelism.md` — the
   surrounding spec.

## Goal

Two llama.cpp processes on ONE machine (127.0.0.1, different ports) split a model
into a head band and a tail band, pass hidden states over TCP, and produce output
**identical to the same model run as a single process**.

This is the correctness gate for the entire cross-machine design. If the composed
pipeline equals the monolithic model bit-exactly on CPU with F32 on the wire, the
layer-boundary semantics are definitionally correct and the cross-machine step
(Phase 3) becomes a transport change with no new semantics.

Cross-machine is explicitly NOT in this task. Localhost only.

## Step 0 — prerequisite

`GpuTransport::init` is reported to have a 4-parameter definition
(`src/weight-pager/wp-gpu-transport.cpp:36`) against a 3-argument call somewhere,
breaking CPU-only builds. **Verify this before assuming it.** If a CPU-only
configure+build fails on it, fix it minimally (one line) and note it. If it builds
clean, say so — the report that it was broken may be stale.

The loopback test needs a working CPU build, so this gates everything else.

## Deliverables

### 1. `src/pipeline/pipe-transport.{h,cpp}`

Lift ONLY the socket plumbing from `ggml/src/ggml-rpc/transport.cpp` (`socket_t`
create/connect/listen, and the `send_data`/`recv_data` loop-until-complete helpers
around transport.cpp:462-556).

- **Do NOT link ggml-rpc.** Do NOT reuse the RPC protocol — no remote op dispatch,
  no tensor addressing, no ggml types on the wire.
- Frame header is the fixed 32 bytes from the design doc (magic `0x4C4C5050`,
  version, type, flags, seq_id, length), little-endian.
- Short reads and short writes must loop to completion. A partial frame is a
  protocol error, never a silently truncated tensor.
- Validate `length` against a maximum before allocating. A peer that claims a
  4 GB payload must be rejected, not honored with a 4 GB allocation.

### 2. `src/pipeline/pipe-protocol.{h,cpp}`

Frame encode/decode for the six frame types in the design doc: HELLO, FWD_REQ,
FWD_RESP, TOKEN, ERROR, PING/PONG.

HELLO validation is the important part: on connect, both sides exchange HELLO and
validate the combined stage set with `llama_pipeline_validate_stages()`. Bands must
be contiguous, non-overlapping, and cover `[0, n_layer-1]`; `n_layer`, `n_embd`, and
`hidden_type` must match. Any mismatch sends PIPE_ERROR and closes. **A pipeline
that produces output from an inconsistent stage set is the worst outcome
available** — it runs, it is fast, and it is wrong. Fail at connect.

### 3. `tools/pipeline/` — the stage driver

Per the design doc's "Stage driver" section:

- **Tail/middle process**: loads its stage, listens on a port. Per FWD_REQ, builds
  a `llama_batch` with `embd` = received hidden, pos/seq_id from the frame, all
  tokens marked as outputs, calls `llama_decode`. Middle reads the embedding output
  and sends FWD_RESP. Tail reads logits, runs its sampler chain, sends PIPE_TOKEN.
- **Head process** (the driver): embeds and runs its own band, reads `t_embd` at the
  band boundary, sends FWD_REQ, blocks on PIPE_TOKEN, appends, repeats.
- Middle/tail MUST run with `--no-warmup`. A token-based warmup decode on a stage
  without `token_embd` fails in `llm_graph_input_hidden::set_input` **by design** —
  that guard is correct; do not weaken it to make warmup pass.
- KV stays local to each stage. No context shifting, no KV migration (Phase 3+).

F32 on the wire for this task. F16 is negotiable in HELLO per the design but is not
to be enabled or tuned here — F32 is what makes the correctness gate exact.

### 4. Tests

**Unit (no model, no inference):**
- Frame round-trip for every type, including empty and maximum-size payloads.
- Truncated frame, bad magic, wrong version, oversized `length` → clean error, no
  allocation, no crash.
- HELLO mismatch matrix: gap between bands, overlapping bands, `n_embd` mismatch,
  `n_layer` mismatch, `hidden_type` mismatch → each must be rejected at connect.
- Endianness: encode/decode must be explicit, not `memcpy` of a struct.

**Loopback correctness harness — WRITE IT, DO NOT RUN IT (see Constraints):**
A script that, given a model path and a split point K:
1. Splits the model into head `[0, K-1]` and tail `[K, n_layer-1]` with
   `tools/wp-stage-split`.
2. Runs the two stage processes on 127.0.0.1.
3. Runs the SAME model as a single process on CPU.
4. Asserts identical prompt logits on the first token, and an identical greedy
   token sequence for N tokens at a fixed seed.

Greedy + fixed seed makes this exact rather than statistical. **Do not soften this
to a similarity threshold or a perplexity comparison.** If the outputs differ, that
is the finding and it must be reported as a failure, not tuned away.

Suggested models for when it is run (do not run them yourself):
`~/models/E2Rank-0.6B.Q8_0.gguf` (639 MB dense — fast gate), then
`~/models/LFM2.5-8B-A1B-Q8_0.gguf` (9 GB MoE — covers the expert path, which is the
shape GLM-5.2 actually has). Run with weight paging OFF first to isolate the
protocol from the pager.

## Constraints — these are hard

- **Do NOT run any model, any inference, `llama-cli`/`llama-server`/
  `llama-perplexity`, or any GPU work.** Not even briefly, not even on CPU, not
  even "just to check". Write the harness; a human runs it. This is a standing
  fleet rule and it is not negotiable.
- **Do NOT touch a GPU.** No `rocm-smi` beyond read-only status, no HIP builds.
  A CPU-only configure+build is fine and is what you should verify against.
- **Do NOT commit, stash, revert, `git checkout`, `git reset`, or `git add -A`.**
  The tree has uncommitted work from other agents. Leave your changes in the
  working tree; a human stages them by explicit path.
- **Do NOT run `npx gitnexus analyze`** or any gitnexus tooling, regardless of what
  the repo CLAUDE.md says.
- **Do NOT restart any service.** `llama-router.service` is live on this box.
- ASCII only in code and comments.

## Invariants

- **A single process with no band set must behave EXACTLY as it does today.** Every
  call site checks `llama_pipeline_band_enabled()` first and takes the legacy path
  when false. This is the regression that matters most; the non-pipeline path is
  what currently works.
- Absolute layer indices are preserved end to end. `blk.57.*` is layer 57 on every
  stage. `hparams.n_layer` stays global; bands never renumber.
- Sampling happens ONLY on the tail. The head applies no sampling of its own.
- One in-flight FWD_REQ per seq_id. Blocking TCP; backpressure is implicit.
- Disconnect mid-request drops the seq_id state on both sides and fails the client
  request loudly. **No retry-with-resume** — silently re-running a partial ubatch
  risks KV divergence, which would surface as subtly wrong output rather than an
  error.

## Report back

- What you built, file by file.
- The Step 0 finding: was the CPU build actually broken, and what did you change.
- Every place you deviated from the design doc, and why.
- Anything you could not verify because you were not permitted to run a model —
  list it explicitly so the human knows what the harness has and has not proven.
  Do not describe untested code as working.
