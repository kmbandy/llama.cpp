# llama-pipeline -- Phase 2 cross-machine pipeline stage driver

One process = one stage = one full llama.cpp instance, joined at a layer
boundary over TCP. See `docs/dev/2026-07-28-pipeline-protocol-design.md` (the
contract) and `docs/dev/2026-07-29-phase2-loopback-spec.md` (this task).

This is the LOOPBACK driver: both stages run on 127.0.0.1 on different ports.
F32 on the wire. Cross-machine (Phase 3) is the same protocol, not built here.

## Roles (derived from the resolved band, not a flag)

- head  (first == 0):            the driver. Embeds, runs its band, reads t_embd
                                 at the boundary, sends FWD_REQ, blocks on
                                 PIPE_TOKEN, appends, repeats. Owns the client
                                 session and the prompt.
- middle:                        hidden in, hidden out (FWD_REQ -> FWD_RESP).
                                 Not exercised in the 2-stage loopback.
- tail  (last == n_layer-1):     hidden in, logits, samples, PIPE_TOKEN.
                                 Sampling happens ONLY here.

## Build

The tool links `llama-common` and `llama` and compiles
`src/pipeline/pipe-transport.cpp` + `src/pipeline/pipe-protocol.cpp` directly
(those are deliberately NOT part of the core `llama` library). Target:
`llama-pipeline`.

## Run (2-stage loopback)

    # terminal 1: tail (listens). --no-warmup is mandatory: a token warmup on a
    # stage without token_embd fails in llm_graph_input_hidden::set_input BY
    # DESIGN. Do not weaken that guard.
    build/bin/llama-pipeline -m tail.gguf \
        --pipeline-listen 127.0.0.1:9911 --no-warmup

    # terminal 2: head (drives)
    build/bin/llama-pipeline -m head.gguf \
        --pipeline-peer 127.0.0.1:9911 \
        -p "The meaning of life is" -n 32 --seed 0 --temp 0

The stage band comes from the stage GGUF's `pipeline.layer_first/last` metadata
(written by `tools/wp-stage-split`) or an explicit `--pipeline-layers FIRST-LAST`.

## Correctness gate

`loopback-test.sh` splits a model at K, runs the 2-stage pipeline and the same
model as a single process, and asserts an identical greedy token sequence at a
fixed seed. It is WRITTEN, NOT RUN, by the implementing agent; a human runs it:

    tools/pipeline/loopback-test.sh ~/models/E2Rank-0.6B.Q8_0.gguf <K> 32

## Known scope limits (Phase 2)

- Single sequence (n_seqs fixed to 1). One in-flight FWD_REQ per seq_id.
- Sampler params reach the tail via identical CLI flags, NOT the design's HELLO
  sampler-state extension (that extension is not implemented in the loopback).
- Middle stages (N>2) are not wired end to end (a middle would need both
  --pipeline-listen and --pipeline-peer; the driver handles one link).
- No retry-with-resume: a disconnect mid-request drops seq state and fails the
  request loudly, by design.
