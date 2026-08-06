# DSpark-driven expert prefetch — pickup brief

> **SUPERSEDED for state and next actions by
> [`2026-08-06-prefetch-morning-brief.md`](2026-08-06-prefetch-morning-brief.md).**
> The work described here as "to build" was built the same night and arm 1 has
> run. What remains live in THIS document is the *reasoning*: why the previous
> attempts failed (§4), why DSpark is different (§5), and the kill criteria (§6,
> as amended). Read this for why; read the morning brief for where things stand.

**Written 2026-08-05 evening, immediately before starting the work.** Everything
below is measured on this fleet unless it says otherwise.

---

## 1. State at handoff

Three commits landed tonight on `master` (local, unpushed):

```
a2d3d585a  docs: version the DS4 harness and record the config of record
a8dc3c9fc  wp-expert-worker: overlap expert compute and page-in reads
c42654ea0  expert-dispatch: send each worker only the activation rows it needs
```

Ten pre-existing dirty files (Aug 3–4 WIP: `speculative.cpp`, `llama-context.cpp`,
the kv-cache set, `deepseek4.cpp`, `server-context.cpp`, `fattn-common.cuh`) are
**deliberately untouched** and still uncommitted. The harness comment block notes
that the `llama_kv_cache_indexer_type` fix among them is load-bearing above ~128K
context.

### Config of record

Harness defaults (`docs/dev/harness-2026-08-05-ds4_full.sh`, md5
`bb6d13cd9156eb878c8690c503c22c30`, byte-identical to the live copy at
`~/.claude/jobs/87d16c2e/tmp/ds4_full.sh` on mad-lab-2026):

KV f16/f16 · CTX=8192 · NPRED=256 · UBATCH=2048 · SPEC=1 · SPEC_CONF=0.99 ·
DSPARK_HOST=CPU · **DSPARK_OMP=8** · DSPARK_TAP=1 · slots 550/550/2200 ·
VKSPLIT=1048576 · **KEEPALIVE=100** · PROMPT=prose739.txt (663 tokens)

Five defaults live in the **C++** and the harness cannot set them. A bare run is
only the config of record if the *binaries* carry them:

| flag | file | default |
|---|---|---|
| `WP_DISPATCH_GATHER` | `src/pipeline/pipe-expert-dispatcher.cpp` | 1 |
| `WP_DISPATCH_GATHER_MAX_FRAC` | same | 0.90 |
| `WP_EXPERT_COMPUTE_CHUNKS` | `tools/wp-expert-worker/wp-expert-worker.cpp` | 4 |
| `WP_EXPERT_READ_STRIPES` | same | 4 |
| `WP_EXPERT_STRIPE_MAX_PAGEINS` | same | 4 |

Verify before trusting any number, and check both build dirs are newer than the
sources:

```
grep -c WP_EXPERT_READ_STRIPES     tools/wp-expert-worker/wp-expert-worker.cpp
grep -c WP_DISPATCH_GATHER_MAX_FRAC src/pipeline/pipe-expert-dispatcher.cpp
```

### Where the time goes now

Spine dispatch, config of record:

| | baseline | after tonight |
|---|---|---|
| PREFILL total | 26.36 s | 22.41 s (−15.0%) |
| decode-side total | 76.48 s | 69.01 s (−9.8%) |

Decode is **80% dispatch**, and dispatch is **98% `wait`**. Per-layer decode wait
is 5.45 ms against a slowest-worker service of ~3.1 ms; the ~2.3 ms difference is
cross-machine WireGuard RTT and is not addressable in software.

---

## 2. What is closed, and why — do not reopen without new evidence

- **Prefill reads.** The RX 480 does 1381 expert references, 29 resident, **1352
  page-ins, 16.83 GiB = exactly 12.75 MB per page-in**. Every page is read
  **exactly once**; zero re-reads, zero amplification. Combined with the shared
  SN750 sitting at its ceiling (4.25 GB/s combined demand vs a 3.94 GB/s gen3 x4
  link), prefill's read leg has no software lever left.
- **Quantization.** DS4-Flash routed experts are **already MXFP4 QAT natively**
  (~4.25 bpv, 13.37 MB/group). There is no representation change that reduces I/O.
- **Socket tuning.** `TCP_NODELAY` already set at all three sites;
  `PIPE_MAX_CHUNK_SIZE` is 64 MiB so a 10.8 MB partial is one `send()`;
  `tcp_wmem` autotunes to 4 MiB against a ~75 KB BDP.
- **`WP_EXPERT_GATHER_MIN_TOKENS=2`.** Broke determinism (0.80952 vs 0.84286 on
  an identical config) and gave no speedup.
- **The MXFP4 Vulkan shader.** Isolated, the RX 480's shader is *faster* per
  expert than CUDA's (115 vs 145 µs). The 6× in-worker divergence is manufactured
  by how the worker drives Vulkan. Leading untested suspect: H2D serializing
  against compute on a single Vulkan queue.

---

## 3. The prize

**The drive is ~78% idle during decode.** The RX 480's own read activity is
**10.3% duty** over the decode window (10.43 s of reads in 101 s); both 2026
workers together are 21.6%. That idle time exists because layer N+1's expert set
is not knowable until layer N's router runs — a hard sequential dependency.

Prefetch is the only thing that can fill it. Nothing else on the board is within
an order of magnitude.

---

## 4. Why previous attempts failed — the honest list

This is the part that matters, because the fear of repeating them is reasonable.

1. **Cross-layer speculative prefetch (M=2…48 sweep).** Net loss at *every*
   width. Cost 2.7–3.1× the bytes via **pool pollution**, with a large fixed term
   (~108 GB) dominating the marginal per-M term 4:1. A predictor at **0.973
   precision@rank-1 still lost** — the recorded conclusion was "predictor quality
   is irrelevant to the outcome." Worse, at K≥2 / M≥8 it was
   **correctness-breaking**: nondeterministic corrupted output under greedy
   decode, caught only by capturing decoded text.
   **Root cause: no lead time.** Predicting within the current token is a 1–2
   layer, sub-10 ms horizon, which cannot hide a ~5 ms cold read. Only **23
   speculative pages** were issued across 128 tokens × 43 layers.
2. **MTP-as-prefetch (depth transfer).** Only 2 of 40 softmax layers cleared the
   0.60 gate. Independent of scoring function.
3. **The DFlash adapter.** 0.14 recall@6. Died.
4. **The `tid2eid` oracle is structurally disconnected on this topology.**
   `deepseek4.cpp:261` marks routed experts `TENSOR_SKIP | TENSOR_NOT_REQUIRED`
   under cross-machine dispatch, so the **spine's pager catalog is empty** and
   `prefetch_hot_experts` / `collect_tid2eid_pages_` resolve to nothing and
   submit nothing. Meanwhile `wp-expert-worker.cpp` has **zero** occurrences of
   `tid2eid` or `prefetch`. *The process with the oracle pages nothing; the
   process that pages has no oracle.*

**Consequence: (1) does not transfer to this fleet.** Those refutations were
measured on the *in-process* pager (and partly on Laguna, which has no `tid2eid`
tensor at all). On the cross-machine topology prefetch has never actually been
tested — it has been measured on a path that submits nothing. That is the third
time in this project a technique was nearly retired on the strength of a broken
build.

---

## 5. Why DSpark is a genuinely different shot

- **A draft block *is* the lead time.** That is precisely what attempt (1)
  lacked. DSpark currently runs at acceptance 0.84286, mean accepted length 1.86,
  block size 5 — so there are real token IDs available several steps ahead.
- **The hash layers need no prediction at all.** `deepseek4.cpp:1808`: for
  `il < dsv4_hash_layer_count` (= 3 on DS4-Flash, layers 0–2) expert selection is
  `ggml_get_rows(ffn_gate_tid2eid, inp_tokens)` — **a pure token-ID lookup with
  no router**. Zero prediction error, no confidence gate needed. Ceiling ~12.5%
  of page-ins (our own 2026-07-09 figure).
- **The bigger prize is PREFILL, not decode.** Prefill already has every token,
  so for the hash layers the *entire* expert schedule is computable before the
  first read. Sort it, dedupe it, stream it in order — that converts prefill from
  a demand-paged random walk into a sequential stream, the only regime where a
  drive's rated bandwidth is reachable.
- **Tonight's work left the right machinery in place.** The worker now has
  `complete_batch_upto()` and a resumable drain (`drain_one_read`), plus striped
  reads. A warm path can reuse that rather than inventing a second read path.

### Minimal correct design

The spine's empty catalog blocks resolving expert→**pages**, but *not*
token→**expert IDs**, because `register_tid2eid_host` (`llama.cpp:501`) hands the
spine the table. So:

> **Spine** computes `tid2eid` for the upcoming draft tokens → sends **expert
> IDs** as a prefetch hint in the dispatch preamble → **each worker** resolves
> those IDs against its own shard and warms its own `ExpertSlotPool`.

No table duplication, no pool-to-pool transport, the spine's catalog never in the
path. It is a protocol addition plus a worker-side warm path.

**One premise to re-check in source before building:** the dispatch request is
`{layer, n_tokens, assignments(expert_id + per-token weights), activations}` —
there are **no token IDs on the wire today**. An external suggestion that "the
spine already sends token IDs" was wrong on exactly this point.

---

## 6. Pre-registered kill criteria — agree these BEFORE the first run

Attempt (1) burned a large sweep before its failure mode was understood. Write
the falsifiers down first this time.

- **The mechanism counter is `n_pagein`, not tok/s.** If prefetch does not reduce
  page-ins, it did nothing, regardless of what throughput says. Tonight
  demonstrated four separate times that end-to-end tok/s gets the answer wrong on
  this rig; per-leg dispatch numbers are the reliable metric.
  **AMENDED 2026-08-05, before the first run: this rule is right for DECODE and
  wrong for PREFILL.** At UBATCH=2048 with `n_expert_used=8`, the hash-layer
  expert union over 2048 tokens covers most of the 256 experts on all three
  layers — the same pages get read either way. The prefill win is not *fewer*
  page-ins, it is issuing them *earlier and in ascending order*, which is the
  only regime where the drive's sequential bandwidth is reachable. So prefill's
  pre-registration is: **page-ins flat, `ns_wait` down, device utilization % up.**
  Judging the prefill arm by `n_pagein` would score a success as a failure.
- **Read-amplification gate (the attempt-1 failure mode).** Track `bytes_read`
  alongside `n_pagein`. If bytes rise while page-ins fall by less, that is pool
  pollution — **stop**, do not tune the predictor. Attempt (1) proved predictor
  quality does not rescue this.
- **Determinism gate.** Draft acceptance must stay **exactly 0.84286 (59/70),
  mean len 1.86**. Attempt (1) was correctness-breaking at K≥2 and it was only
  caught by inspecting decoded text.
- **Utilization, not queue depth.** The 07-31 finding stands: the drive is at
  QD 11–14 already and busy only 30–40% of the time. If a prefetch build does not
  move **device utilization %** off that floor, it did not do what it claims.
- **Scope the first cut to the hash layers only** (layers 0–2, pure token-ID
  lookup, zero prediction error). If prefetch cannot win *there* — where the
  oracle is exact and free — it will not win anywhere, and that is a cheap,
  decisive answer rather than another wide sweep.

## 7. Measurement discipline that worked tonight

- Attribute **per leg**, not by throughput. Each change should move its own leg
  with the others flat; that is a within-run control and survives host load.
- **Interleave arms, never block them.** Load on mad-lab-main swings 5.8–17.9
  during ordinary desktop use.
- **Sample and print load per rep**, beside the numbers, so contamination is
  visible in the data rather than discovered afterwards.
- mad-lab-main is a **desktop in active use**, and that is a legitimate operating
  condition, not a disqualifier. Host load inflates every phase at once — which
  is exactly why a clean per-leg group separation is credible where a tok/s delta
  is not.
