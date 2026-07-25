# Two-process layer split across machines — design

**Date:** 2026-07-25 · **Repo:** `~/GitHub/llama.cpp` on mad-lab-main, tip `a0e7aec25`
**Goal (kmbandy, standing):** run DeepSeek-V4-Flash across all four GPUs on both
machines with ~8 GB of host RAM per machine, weight-paging on both sides. Throughput is
explicitly secondary.

## 1. Why two processes, and a correction to the record

The roadmap and my own earlier statements assert *"llama.cpp has no pipeline parallelism
and RPC cannot express it."* **Both halves of that are wrong**, and the real blockers are
different:

- **Pipeline parallelism exists and is live.** `llama-cparams.h:48` `pipeline_parallel`,
  gated `llama-context.cpp:395-401`, implemented via `n_copies` + per-backend events
  (`ggml-backend.cpp:2012-2013`) so graph *n+1*'s early splits overlap graph *n*'s later
  ones. `docs/multi-gpu.md:25` documents `-sm layer` as exactly this.
- **RPC can express layer placement.** RPC devices register as ordinary devices and are
  sorted to the front of `model->devices` (`llama.cpp:614`), and contiguous per-device
  layer ranges already exist — `llama-model.cpp:1603-1622` derives `layer_gpu` from the
  cumulative `--tensor-split`, monotone in `il`. `--rpc host:port -sm layer -ts 70,30`
  is already valid syntax.

The actual blockers, in descending severity:

1. **The weight pager cannot span devices, by design and by implementation.**
   `llama.cpp:145-155` throws outright if paged weights span two GPU buffer types. The
   pool is a *single* `ggml_backend_buffer` on one device (`wp-pool.h:12-19`). The
   transport is raw HIP throughout — `hipSetDevice` in `wp-gpu-transport.cpp`,
   `hipMemcpyAsync` and `ggml_cuda_set_routed_expert_ptrs` in `wp-eval-cb.cpp` — plus a
   dma_buf export of the VRAM pool for NVMe→VRAM P2P (`wp-file-io-p2p.cpp:475,726`).
   There is no device pointer behind a socket to `hipSetDevice` to, and no dma_buf to
   export. Worse, `device_idx` is parsed out of the *device name string*
   (`llama.cpp:158-170`), so an RPC device named `"RPC0"` yields a bogus index that is
   then handed to `hipSetDevice`.
2. **Cross-machine execution through RPC would be fully serialised anyway.** RPC reports
   `caps.async=false, caps.events=false` with `event_new=NULL`
   (`ggml-rpc.cpp:1792-1802,1837`), which trips `llama-context.cpp:412-430` and forces
   `pipeline_parallel=false`, `n_copies=1`. With `cpy_tensor_async=NULL` every split
   boundary takes the `ggml_backend_synchronize` path on both sides — full round-trip
   latency per layer boundary, zero overlap.
3. Our fork already closes this door deliberately, twice:
   `llama-context.cpp:406-410` disables pipeline parallelism whenever
   `wp_pager != nullptr`; `llama-model.cpp:1540-1543` sets `has_tensor_overrides = true`
   specifically to defeat the gate, because otherwise `graph_reserve` attempts a ~90 GiB
   compute buffer.

**Two independent pagers therefore require two processes.** The split cannot live inside
`ggml_backend_sched`. It is two `llama_context`s with an activation channel between them.

## 2. The observation that makes this easy

**Decode is ~3 tok/s — about 333 ms per token — and the boundary payload is kilobytes.**

A single hidden-state slice at a layer boundary is `n_embd` values per token (times any
hyper-connection stream multiplier — see §6). At `n_embd = 4096` in f16 that is 8 KB per
token per stream. Even at a 4× stream multiplier it is 32 KB. Over a LAN with sub-
millisecond RTT, one round trip per token against a 333 ms token budget is **well under
1% overhead.**

The consequence is worth stating plainly because it removes most of the difficulty:
**cross-machine pipelining is unnecessary for single-stream decode.** A simple
synchronous request/response channel suffices. We do not need to fix RPC's async/events
gap, do not need `n_copies > 1` across the boundary, and do not need to touch
`ggml_backend_sched` at all.

**Prefill is the transfer that actually costs.** A 4096-token prompt at 4096 × f16 is
~34 MB per stream (~134 MB at 4×). One-time per request, chunkable, and still seconds at
worst. Size it, don't fear it.

## 3. Topology

Two stages, contiguous layer ranges, connected by one channel:

- **Stage 0 — mad-lab-main.** Token embeddings, layers `0..N-1`, its own weight pager
  (paging from main's NVMe), its own KV cache for those layers, both local GPUs. R9700
  as paging device, 6900 XT as resident/dense per the existing router roles.
- **Stage 1 — mad-lab-2026.** Layers `N..43`, the output head, its own pager (paging
  from 2026's NVMe), its own KV for those layers, GTX 1070 + RX 480.

Per the roadmap's ~70/30 storage split, `N ≈ 31`, putting ~13 layers on 2026.

Per decode step: stage 0 computes its layers and sends the boundary activation; stage 1
computes the rest, applies the head, samples, and returns **the sampled token id** — four
bytes, not a logits vector. Stage 0 embeds that token for the next step. One round trip
per token.

**Which end owns sampling is a real decision, not an implementation detail.** Putting the
head and sampler on stage 1 minimises return traffic but places sampling, grammar and
penalties on the weaker machine and splits the sampler state away from the API surface.
The alternative — stage 1 returns logits (`n_vocab` = 129280 × f16 ≈ 259 KB/token,
still trivial) and stage 0 samples — keeps all user-facing state in one place. **I
recommend stage 0 samples**, on the grounds that 259 KB/token is free at these speeds and
keeping the server's sampler, grammar and slot state undivided is worth far more than the
bandwidth.

## 4. The three genuinely new pieces of work

### 4a. Layer-range graph construction

There is **no layer-subset concept for the model today** — `il_start`/`il_end` exist only
for control-vector application (`llama-adapter.cpp:99-122`,
`llama-context.cpp:1344-1348`). This is the substantial piece.

Requirements:
- A parameter selecting a contiguous layer range for this process.
- The graph builder emits only layers in range.
- For a non-first stage, the graph's **input** is the boundary activation tensor rather
  than the token-embedding lookup.
- For a non-last stage, the graph's **output** is the boundary activation rather than
  logits/head.
- The KV cache is sized for the owned layers only.
- `graph_reserve` and compute-buffer sizing must reflect the reduced range — note the
  existing ~90 GiB compute-buffer trap at `llama-model.cpp:1540-1543` as evidence this
  path is sensitive to getting sizes wrong.

**Reuse:** `llama_context::extract_layer_inputs` and `llm_graph_result::get_layer_inp(il)`
already exist (`llama-context.cpp:2294-2302`), built to feed the DFlash drafter a
mid-stack residual. That is precisely the tap a stage boundary needs, and it is the
natural place to start rather than inventing a new extraction path.

### 4b. Partial model loading

Each stage should load only the tensors for its own layers. Two candidate approaches, and
the choice should be made from measurement rather than taste:

- **Split the GGUF shards physically**, so each machine's files contain only its layers'
  tensors. Matches the roadmap's "shave off some of the weight and place them over
  there", halves the bytes each machine stores, and is the only option that fits 2026's
  smaller drive. Requires a splitting tool and metadata that survives it.
- **Both stages read the same full model metadata** but materialise only their own range.
  Simpler to implement, but every machine must store all 149 GB — which defeats the
  storage motivation entirely.

The loader currently expects each layer's tensors to exist, though `TENSOR_NOT_REQUIRED`
is already used in 52 places, so optional-tensor machinery exists to build on.

### 4c. The activation channel

A purpose-built stage-to-stage channel. Deliberately **not** RPC-as-a-device — that path
brings the whole `supports_op`/async/events problem set for no benefit here.

- Synchronous request/response, one round trip per decode step. Justified by §2.
- Payload: boundary activation (stage 0 → 1), logits or token id back (see §3).
- Must carry enough context for stage 1 to place work correctly: sequence id, position,
  and whatever the KV cache needs to keep the two stages' caches consistent.
- **Sequence and cache lifecycle is the subtle part.** KV eviction, sequence removal,
  context shift, and prompt-cache reuse all have to happen on *both* stages consistently.
  A stage-1 cache that diverges from stage 0's produces silently wrong output rather than
  an error. This deserves more design attention than the transport does.
- Failure semantics: if a stage dies mid-request, the survivor must fail the request
  cleanly rather than hang or serve garbage.

## 5. Phasing — de-risk before touching the big model

**Phase 0: prove the mechanism on one machine, small model, localhost.**
Two processes, a small dense model, split at a layer boundary, no weight paging, no
network. This exercises layer-range graphs, the boundary tap, and the channel while
keeping every variable else fixed. **Success criterion: token-for-token identical output
versus the same model unsplit, greedy sampling, same seed.** Nothing else proves the
boundary is mathematically transparent, and everything downstream depends on that.

**Phase 1: same thing across the two machines**, still small model, still no paging.
Adds real network latency and the failure modes that come with it. Measures the actual
per-token round-trip cost against the §2 estimate.

**Phase 2: enable weight paging independently on each stage.** Each has its own pager and
its own NVMe. This is where the 8 GB-per-machine RAM budget gets tested for real.

**Phase 3: DS4-Flash across both machines.** Requires the GGUF split (4b) and the
boundary tensor question (6a) resolved.

Phase 0 is where the design gets validated or falsified cheaply. Do not skip it, and do
not start it on DS4.

## 6. Open questions — resolve before building

1. **What exactly crosses the boundary for DS4?** DS4-Flash uses hyper-connections; the
   KG records the DFlash drafter consuming a *4-stream pre-collapse* residual
   (`fc.weight = [4096, 81920] = 5 tap layers × hc_mult(4) × 4096`). If the residual is
   genuinely multi-stream at a layer boundary, the payload is 4× and the cut must land
   where the streams are well-defined. **Note:** the `n_stream` in `llama-graph.cpp:37,56`
   is the KV sequence-stream count (`ubatch.n_seqs_unq`), *not* the hyper-connection
   multiplier — I conflated these initially. `get_hca_plan` / `dsv4_set_comp_inputs` were
   not found in `llama-graph.cpp`. **This is unverified and must be settled from source
   before sizing anything.**
2. **Where does the cut go?** ~70/30 by storage says `N ≈ 31`, but the right boundary may
   be determined by compute balance, KV size, or where the residual is cleanest — not by
   storage alone.
3. **Which end samples?** §3 recommends stage 0. Confirm.
4. **How is the GGUF split?** Tooling, metadata, and whether shard 4 (18.3 GB, 91%
   compressible, likely mostly inert) can simply live wherever is convenient.
5. **What owns the user-facing API?** Presumably stage 0 runs the server and stage 1 is
   headless. Confirm, and decide how stage 1 is launched and supervised.
6. **Does the eval-callback path interact badly?** With `wp_pager` present the scheduler
   switches to a per-node loop with `ggml_backend_synchronize` after each node group
   (`ggml-backend.cpp:1826-1857`). This is *probably* moot in a two-process design since
   no split boundary crosses machines inside one graph — but it was flagged as the least
   certain part of the RPC reading and should be confirmed rather than assumed.

## 7. Out of scope

- Fixing RPC's `caps.async`/`events` gap or `cpy_tensor_async`. Not needed here.
- Cross-machine pipelining / `n_copies > 1` across the boundary. Not needed for
  single-stream decode (§2). It *would* be needed for concurrent multi-request serving —
  a later question.
- `ggml_backend_sched` changes of any kind.
- Prefetch and the 5.29 in-flight ceiling. Orthogonal, and tracked separately.

## 8. Independent of this work

**`ggml_backend_rpc_device_supports_op` returns `true` unconditionally**
(`ggml-rpc.cpp:1822-1827`, `//TODO: call the remote backend and cache the results`). That
is a live correctness bug for anyone using RPC devices, unrelated to the split, and cheap
to fix. Worth doing on its own merits.

Also noted while reading: `RPC_CMD_GRAPH_COMPUTE` is fire-and-forget on the wire
(`ggml-rpc.cpp:717` uses the no-response overload; the server at `:1653-1662` sends
nothing back), so the "no-op because we don't have any async operations" comment at
`:653-654` is stale. There is latent overlap the scheduler does not exploit. Unmeasured —
a lead, not a fact.

## 9. Honest scope assessment

This is a **large** piece of work — layer-range graph construction alone touches model
loading, graph building, KV sizing and buffer reservation, and none of it exists today.
Phase 0 is the cheap part and the part that decides whether the rest is worth starting.

The design's virtue is what it *doesn't* need: no scheduler changes, no RPC protocol
work, no cross-machine pipelining. That falls out of decode being slow enough and the
boundary payload small enough that a synchronous round trip per token is free.
