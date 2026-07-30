# Cross-machine expert dispatch — 4-GPU scheduler design

**Status:** design, not built. Supersedes the layer-band pipeline as the way to
add GPUs. Author: Claude, 2026-07-30. Implementation: terra, staged.

---

## 1. Goal

Use all four GPUs across both machines as **expert compute nodes** for GLM-5.2
(753B, UD-Q2_K_XL), coordinated by a scheduler that assigns each activated
expert to a device by ownership, residency and load — rather than by a fixed
hand-chosen layer partition.

Four workers: R9700 (32 GB, ROCm), RX 6900 XT (16 GB, ROCm), GTX 1070 (8 GB,
CUDA), RX 480 (8 GB, Vulkan). The 480 is a first-class compute node, not a
weight shelf.

---

## 2. Why expert dispatch and not layer bands

Layer bands parallelise the one axis that cannot be parallelised. For a single
token, layer *N* consumes layer *N-1*'s output; splitting layers across machines
adds a hop and no throughput. This is measured, not argued: main 0.655 s +
2026 0.565 s = 1.22 s against a 1.39 s single-process baseline. The 2-stage
cross-machine split ran correctly today and produced **no speedup by
construction**.

MoE is the opposite. GLM-5.2 activates **8 of 256 experts per layer**, and those
8 are mutually independent — genuine parallelism, available 75 times per token.

This is also the fleet's own prior conclusion
(`docs/dev/2026-07-24-rpc-mesh-weight-pager-integration.md` §7):

> remote GPUs must not host the attention island … they should do **expert
> compute**. Expert compute is per-token *parallel* — ship a ~12 KB activation,
> compute, return a partial weighted sum: one round trip — whereas attention is
> sequential per layer.

and it names `docs/dev/2026-07-21-tiered-dual-gpu-expert-feeding-design.md` §7.3
as "the unbuilt piece". This design is that piece, generalised from 2 GPUs on
one machine to 4 GPUs across two.

### The principle that makes a scheduler possible

An expert's **weights are 12.22 MB**; its **activation is ~12 KB**. Three orders
of magnitude. So the scheduler ships *activations to where the weights already
are*, never weights to where compute is free. Every placement decision follows
from that asymmetry.

---

## 3. Measured constraints (not estimates)

| Quantity | Value | Source |
|---|---|---|
| Expert bytes, layers 3–77 | **234.7 GB** | repack manifest |
| Per-expert page | **12.22 MB**, one contiguous read | repack (landed today, `0f0a8bb11`) |
| Experts / layer, active | 256 / **8** | GGUF metadata |
| Routing distribution | sigmoid `noaux_tc`, **near-uniform** | model config |
| Bytes fed per token | **7.33 GB** (8 × 75 × 12.22 MB) | derived |
| NVMe main (SN850X, O_DIRECT) | **6.25 GB/s** | measured 2026-07-24 |
| NVMe 2026 (SN550, O_DIRECT) | **3.08 GB/s** | measured 2026-07-24 |
| Inter-machine link | **104 MB/s**, RTT **0.5–0.6 ms** | measured 2026-07-24 |
| RAM, main / 2026 | 15 GB total, **9 / 12 GB available** | measured today |
| Disk free, main / 2026 | **98 GB** / 36 GB (**113 GB** if the 77 GB tail stage is removed) | measured today |

Two constraints dominate and must shape everything:

1. **2026 cannot store the whole expert set.** 234.7 GB against 113 GB best case.
   So a 2026 GPU can only compute experts whose weights live on 2026.
2. **Both machines are RAM-poor** (15 GB, shared with desktop/MCP/dashboard).
   The victim tier is bounded to single-digit GB, not tens.

---

## 4. Architecture

### 4.1 Roles

- **Attention spine — R9700.** All dense/attention/KV (14.5 GB non-expert) plus
  the router. Sequential and latency-critical, so it sits on the fastest local
  card and never moves. This is the July finding applied directly.
- **Expert workers — all four GPUs**, including the R9700 with its remaining
  VRAM. Each owns an LRU pool of expert pages and computes the experts assigned
  to it.
- **Scheduler** — runs in the spine process. Per MoE layer, partitions the
  router's top-8 across workers and reduces their partial sums.

### 4.2 Storage sharding — by expert index, not by layer

Shard the expert set **by expert index across all layers**, not by layer range.
With near-uniform routing this gives every machine work in *every* layer;
sharding by layer would idle 2026 for 52 of 75 layers.

Split proportional to measured feed bandwidth — 2026's share is
3.08 / (6.25 + 3.08) = **33.0%**:

| Machine | Experts | Storage | Expected experts/layer |
|---|---|---|---|
| 2026 | indices **0–84** (85 of 256) | **77.9 GB** (fits 113 GB) | 2.64 of 8 |
| main | indices **85–255** (171) | 156.8 GB (already has the full repack) | 5.36 of 8 |

This balances the per-token feed almost exactly:

```
main : 7.33 GB x 67.0% / 6.25 GB/s = 786 ms
2026 : 7.33 GB x 33.0% / 3.08 GB/s = 786 ms      -> both finish together
```

**Decision required (see §7):** 2026 must give up the 77 GB tail stage file to
hold its shard. That file is the only local copy; it can be re-cut from main.

### 4.3 Work unit and wire protocol

One **dispatch** per (layer, worker), not per expert — batching a worker's
experts into a single message is what keeps the round-trip count at 75/token
instead of 600.

Request carries: layer index, sequence/step id, token count, the list of
(expert_id, per-token gate weight) pairs assigned to that worker, and the
activation block [n_tokens × n_embd].

Response carries: the same ids, and the **partial weighted sum**
Σ over assigned experts of w_e · FFN_e(x), shaped [n_tokens × n_embd].

Design the frames **token-batched from the start** (n_tokens ≥ 1 with per-token
weights, zero where a token did not route to that expert) even though stage 1
runs a single token. Retrofitting batching into a scalar protocol is the kind of
rework that makes microbatching a rewrite instead of a flag.

Wire cost per token: 75 layers × ~25 KB round trip ≈ 1.8 MB → ~18 ms at
104 MB/s, plus 75 × 0.5 ms RTT ≈ 37 ms. Total ~55 ms **against 786 ms of local
expert compute it overlaps with** — hidden, provided dispatch is issued before
local work begins (§4.4).

Transport: reuse the existing pipe framing (`pipe-protocol`/`pipe-transport`) —
HELLO, length-prefixed frames, and the cross-machine socket already work and are
exercised. New frame types for dispatch/partial rather than a new transport.

### 4.4 The scheduler — two levels

**Level 1 — machine, static.** Expert index → machine, fixed by the storage
shard. Not a choice at runtime: an expert can only be computed where its weights
are stored.

**Level 2 — GPU within a machine, dynamic.** This is where "whichever GPU is
available" lives. Both GPUs on a machine draw from the same NVMe, so the
machine's feed is the shared constraint and the per-GPU decision is about
**residency and queue depth**:

1. If expert E is already resident in a GPU's pool → assign there (zero feed cost).
2. Else assign to the GPU on that machine with the shallowest queue.
3. Break ties toward the larger pool (more future residency).

Ordering, and it matters: **issue remote dispatches first, then compute local
experts, then collect.** Getting this backwards serialises the link behind local
compute and throws away the overlap that makes the whole design free.

### 4.5 Reduction and numerics

Expert outputs combine additively, so each worker returns a partial sum and the
spine adds them. Summation order will differ from single-device execution;
results will not be bit-identical to a single-GPU reference. That is expected
and must be stated in the gate rather than discovered — see §6.

### 4.6 RAM victim tier — both machines

`HostTier` (`src/weight-pager/wp-host-tier.cpp`) already exists: pinned host
arena, LRU, `store`/`lookup` by page index, `WP_HOST_BUDGET_BYTES`. It is wired
only into the serial `page_in_sync_` fallback; the fast `ensure_batch` P2P path
bypasses it deliberately, because P2P leaves no host copy to cache.

Per the July design, the resolution is that the demand path **only reads** the
cache and an async worker **populates** it. Build both machines' tiers now with
that split, sized to what these RAM-poor boxes allow: **~6 GB main, ~6 GB 2026**,
leaving desktop/MCP/dashboard headroom.

**Set expectations honestly:** with near-uniform routing a 6 GB tier against a
156.8 GB (main) / 77.9 GB (2026) shard is 3.8% / 7.7% of the working set. A prior
LRU simulation on this model measured 19.2% hit against a 14.7% baseline — real
but modest. The tier is in scope because it subtracts NVMe reads and the feed is
the bottleneck; it is not where the 1.5x comes from.

### 4.7 Prefetch plumbing

Structure only, no policy. The populate-side worker of §4.6 *is* the prefetch
seam: give it a queue interface and a pluggable source of "pages to fetch", and
leave the policy a no-op. Prefetch optimisation is explicitly deferred — prior
work refuted cross-layer speculative prefetch for this configuration (a perfect
predictor caps at 1.13x and needs a ~46 GB arena), so no policy should be built
until the system runs and can be measured.

---

## 5. Expected gain — first-order, and honest

Feed-bound, so the gain is the aggregate bandwidth ratio:

```
main alone : 7.33 GB / 6.25 GB/s          = 1173 ms/token
both       : max(786, 786)                =  786 ms/token
                                          -> 1.49x
```

Against today's measured 1.39 s/token, expect **~0.8–0.9 s/token (≈1.5x)** once
dispatch works, before any caching. The RAM tier adds a few percent. That is the
whole first-order case; anything beyond it must be measured, not projected.

**This fleet has a habit of refuting first-order estimates** — the P2P zero-copy
work landed a measured 4x on its mechanism and moved decode throughput by a
median +0.1%. Treat 1.49x as a hypothesis with a gate attached, not a promise.

---

## 6. Risks and gates

| # | Risk | Gate |
|---|---|---|
| R1 | **Vulkan expert compute on the RX 480 is unproven.** Every prior attempt to give the 480 real work produced garbage or crashed. It has never been asked to compute an expert FFN it owns. | **Stage 0, before anything else.** Compute one expert FFN on Vulkan0 and compare against the same expert on ROCm/CUDA within tolerance. If this fails, the 480 is a Vulkan backend bug, and that becomes its own investigation rather than a blocked design. |
| R2 | Partial-sum reduction changes numerics | Gate on coherence + a tolerance bound vs single-device, never bit-identity |
| R3 | Remote dispatch not overlapped → link becomes serial | Assert issue-before-local-compute; measure link idle during local expert compute |
| R4 | 2026 shard eviction thrash (8 GB pools vs 77.9 GB shard) | Report pool hit rate per worker; it is the number that says whether pools are sized right |
| R5 | Losing the tail stage file (§7) | Re-cuttable from main in ~5 min; confirm before deleting |
| R6 | Scheduler starves a worker under skewed routing | Log per-worker experts/layer; near-uniform routing should hold it near 2.64 / 5.36 |

---

## 7. Decisions needed from kmbandy

1. **Delete 2026's 77 GB tail stage** to make room for its 77.9 GB expert shard?
   It is the only local copy; re-cuttable from main. Without this 2026 has 36 GB
   and cannot hold a useful shard.
2. **Shard ratio** — 33/67 by measured NVMe bandwidth, or weight it differently
   (e.g. give 2026 less so its 8 GB pools cache a larger fraction of its shard)?

---

## 8. Build stages

Each stage is independently testable and leaves the tree working. All GPU runs
are executed by Claude, never by an implementation agent.

- **Stage 0 — Vulkan expert-compute gate (R1).** Prove the 480 computes an
  expert FFN correctly. Blocks everything; cheap; do it first.
- **Stage 1 — expert shard builder.** Produce 2026's expert-index shard from the
  repack blobs + its sidecars, with a manifest. Reuses today's blob format and
  loader. Verifiable offline against the source blobs.
- **Stage 2 — worker service.** A process that owns a device, loads its shard's
  pager catalog, accepts dispatch frames, computes Σ w_e·FFN_e(x), returns the
  partial. Testable single-machine, one worker, against an in-process reference.
- **Stage 3 — scheduler + reduction in the spine.** Router top-8 → partition →
  issue remote first → compute local → reduce. Correctness gate vs single-process.
- **Stage 4 — 4 workers, cross-machine.** All GPUs, both machines. Coherence
  gate, then throughput measurement against the 1.49x hypothesis.
- **Stage 5 — RAM victim tier on both machines**, read-on-demand /
  populate-async, with the prefetch queue seam left inert.

---

## 9. Non-goals

- Prefetch *policy* (plumbing only — §4.7)
- Speculation / MTP (separate work; the repack already covers `blk.78`)
- Replacing the weight pager, the repack format, or the RPC mesh
- Bit-identical output vs single-device (§4.5)
- Keeping the layer-band pipeline as a scaling mechanism — it stays as a
  correctness harness and for spanning storage, not for adding GPUs
