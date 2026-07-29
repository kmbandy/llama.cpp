# Phase 2 design: the cross-machine pipeline protocol

Status: DESIGN (Phase 1 and 1b implemented alongside this document).
Spec: docs/superpowers/specs/2026-07-28-cross-machine-pipeline-parallelism.md.

## Scope and expectations (restated honestly)

Single-stream decode only. The stages are strictly sequential: token time is
the SUM of stage times, not the max. Expected gain ~+10% from added cache
capacity, plus the ability to hold a model too large for one box. No
multi-token-in-flight pipelining in this phase.

## Roles

- HEAD: owns layers [0, A], token_embd, its own KV. Produces hidden states.
- MIDDLE (N>2 pipelines only): owns [A+1, B], its own KV. Hidden in, hidden out.
- TAIL: owns [B+1, n_layer-1], output_norm + output (lm_head), its own KV.
  Samples. Returns token ids.

One process = one stage = one full llama.cpp instance (llama-server with
--pipeline-layers F-L, or a stage GGUF carrying pipeline.layer_first/last).
The head process is also the pipeline DRIVER: it owns the client session
(chat completion request), the prompt, the sampler parameters, and the loop.

## Wire format

Length-prefixed frames over TCP. All integers little-endian. Text is ASCII.

Every frame:

```
u32  magic        = 0x4C4C5050 ("LLPP")
u32  version      = 1
u32  type         = frame type (below)
u32  flags        = reserved, 0
u64  seq_id       = pipeline request id (see below)
u64  length       = payload bytes following this header
u8[] payload
```

24-byte fixed header, then payload. Frame types:

```
PIPE_HELLO       = 1  (both directions, once per connection)
PIPE_FWD_REQ     = 2  (head -> next stage: run a ubatch)
PIPE_FWD_RESP    = 3  (middle/tail -> caller: result of the ubatch)
PIPE_TOKEN       = 4  (tail -> head: sampled token ids)
PIPE_ERROR       = 5  (any -> caller: fatal, abort the request)
PIPE_PING/PONG   = 6/7 (liveness; also used to measure RTT at connect)
```

### PIPE_HELLO (handshake)

Payload:

```
u32  role          (0=head, 1=middle, 2=tail)
i32  layer_first
i32  layer_last
i32  n_layer       (global)
i32  n_embd
i32  hidden_type   (0 = F32, 1 = F16)
u32  model_hash    (first 4 bytes of the stage GGUF's structural hash)
```

On connect both sides exchange HELLO and validate with the Phase 1 helper
`llama_pipeline_validate_stages()`: bands must be contiguous, non-overlapping,
cover [0, n_layer-1]; n_layer and n_embd must match on both sides;
hidden_type must match. Any mismatch -> PIPE_ERROR + close. This is the
cross-machine enforcement of the "roles consistent" invariant that a single
process cannot check alone.

### PIPE_FWD_REQ (head/middle -> next stage)

```
u32  n_tokens
u32  n_pos_per_embd
i32  pos[n_tokens * n_pos_per_embd]
u32  n_seqs                       (Phase 2 fixed to 1; field reserved)
i32  seq_tokens[n_seqs]           (tokens per sequence; sums to n_tokens)
u8[] hidden, n_embd * n_tokens values, hidden_type as negotiated
```

One FWD_REQ is one ubatch: the same granularity llama_decode already uses.
Prompt processing sends FWD_REQs of n_ubatch tokens; decode sends
1-token FWD_REQs.

seq_id ties the request to a generation session on the driver and lets a
stage reject a stale frame from a previous session after a reconnect.

### PIPE_FWD_RESP (middle -> head)

```
u32  n_tokens
u8[] hidden, n_embd * n_tokens values, hidden_type as negotiated
```

### PIPE_TOKEN (tail -> head)

```
u32  n_tokens        (number of sampled ids, == FWD_REQ.n_tokens for decode)
i32  token_ids[n_tokens]
```

The tail samples with the sampler chain the DRIVER sent at session start
(see "sampler state" below) and returns ids, not logits -- 4 bytes per token
crosses back, per the architecture.

### PIPE_ERROR

```
u32  code
u16  msg_len
u8[] msg
```

Fatal for the seq_id. Driver aborts the client request.

## Precision on the wire

n_embd = 6144. F32 hidden = 24 KB/token; F16 = 12 KB/token.
Default: F32, because it is lossless and 24 KB is nothing against a ~2000 ms
token (12-24 KB over even a 100 Mbit link is single-digit ms). F16 is
negotiated in HELLO (hidden_type=1) and halves the crossing; the conversion
happens once per stage boundary per token, on the CPU, before send. If a
quality regression is ever measured it will show up in perplexity first --
switch it then, not before.

## Transport

Reuse the ggml-rpc SOCKET PLUMBING ONLY: ggml/src/ggml-rpc/transport.cpp's
socket_t (create/connect/listen plus the `send_data`/`recv_data`
loop-until-complete helpers at transport.cpp:462-556) -- lift them into a
small shared `src/pipeline/pipe-transport.{h,cpp}` -- do NOT link ggml-rpc,
and do NOT reuse the RPC protocol: no remote op dispatch, no tensor
addressing, no ggml types on the wire.

Why not RPC semantics, restated: ggml-rpc executes remote ops on
client-allocated tensors; the server has no model, no loader, no catalog, so
there is nothing on the remote side to page. The pipeline protocol moves
UBATCHES and HIDDEN STATES between two full llama.cpp instances.

## Stage driver (per process)

New tool `tools/pipeline/` (or `llama-server --pipeline-peer host:port`):

- TAIL/MIDDLE process: loads its stage, listens on a port. Per FWD_REQ it
  builds a llama_batch with embd = received hidden (F32; if negotiated F16,
  dequantized on receipt), pos/seq_id from the frame, all tokens marked as
  outputs, calls llama_decode. Middle: reads the embedding output
  (cparams.embeddings=true; t_embd covers all n_tokens because out_ids is set
  to all tokens) and sends FWD_RESP. Tail: reads logits, runs its sampler
  chain, sends PIPE_TOKEN.

  Operational note: middle/tail stages MUST run with --no-warmup (or the
  driver-level equivalent). A token-based warmup decode on a stage without
  token_embd fails loudly in llm_graph_input_hidden::set_input by design --
  that is the guard working, but it would abort startup. The head stage
  warms up normally.

- HEAD process (the driver): normal llama-server front end. For each
  generation: embed + run its own band via llama_decode; read t_embd for the
  last band boundary (embeddings=true, all tokens output); send FWD_REQ;
  block on PIPE_TOKEN; append token; repeat. Prompt processing pipelines the
  same way with n_ubatch-sized frames.

- KV and seq management stay LOCAL to each stage: each stage applies the
  positions it is given. seq_rm / context shifting is driven by the head
  re-sending (Phase 2: full recompute on overflow, same as a naive client;
  shifting across a pipeline is Phase 3+).

## Sampler state

Sampling happens ONLY on the tail (it owns lm_head). The driver forwards the
resolved sampler parameters (temperature, top-k/top-p, penalties, seed) in an
extension of HELLO at session start (fixed-size POD block + optional grammar
string frame). The tail rebuilds its sampler chain per seq_id. The head
applies no sampling of its own.

## Failure and backpressure

- Blocking TCP, one in-flight FWD_REQ per seq_id: backpressure is implicit.
- Disconnect mid-request -> both sides drop the seq_id state; the driver
  fails the client request loudly. No retry-with-resume in Phase 2: a
  pipeline that silently re-runs a partial ubatch risks KV divergence.
- A stage that dies between requests is detected by PING at the driver's
  next token; the driver fails fast rather than queuing.

## Loopback-first test plan (verifiable on main alone)

1. Split a small MoE test GGUF (e.g. a Qwen3-Moe-style tiny model, or a
   synthetic glm-dsa-shape file produced by wp-stage-split once GLM-5.2
   lands) into head/tail stage files with wp-stage-split.
2. Run two stage processes on 127.0.0.1, different ports: head owns
   [0, K-1], tail owns [K, n_layer-1].
3. Assert, against the SAME model run as a single process on CPU:
   - identical prompt logits on the first token (F32 wire),
   - identical greedy token sequence for N tokens with a fixed seed.
   Greedy + fixed seed makes the comparison exact, not statistical.
4. HELLO mismatch cases (gap, overlap, n_embd mismatch) must fail at connect
   with PIPE_ERROR, never produce output.

Step 3 is the correctness gate for the whole design: a pipeline whose
composed output equals the monolithic model's output on CPU, bit-exact for
F32, is definitionally correct at the layer-boundary semantics.

## Phase 3 (design only): cross-machine over Tailscale

Same protocol, no changes. main = 100.86.191.92, 2026 = 100.102.191.30.
Measured RTT 0.5-0.6 ms -> ~1 ms/token of crossing against a ~2000 ms token.
Negligible; do not optimise. The one operational addition: bind stages to
the Tailscale interface, and add a --pipeline-psk shared-secret check in
HELLO (HMAC of the HELLO block) so a stage refuses a connection that is not
its peer -- Tailscale ACLs already isolate the nodes, this is only defence
in depth against operator error (wrong port).

## What is explicitly NOT in Phase 2

- Multiple tokens in flight / microbatch pipelining (the 2x path). Out of scope.
- Speculative decoding across the boundary.
- KV migration / context shifting coordination.
- Heterogeneous hidden_type per hop (one negotiated value per pipeline).
- N > 3 stages (protocol supports MIDDLE; untested beyond loopback 2-stage).
