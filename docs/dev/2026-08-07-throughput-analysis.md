# DS4 cross-machine weight paging — throughput analysis & recommendations

**Written** 2026-08-07 night · **Author** Claude · **Scope**: read-only code + ledger
review. No builds, no runs, no GPUs touched. Repo state reviewed: `679ce61e4` +
the 9 pre-existing WIP files (untouched).

Baseline of record: **~15–20 t/s prefill / ~3.9 t/s decode** (code700, banked
stack, SPEC_NMAX=7 SPEC_CONF=0.4, STRIPEPAR=1, COALESCE=1). Targets: prefill
80–100, decode 10–20.

This document is organized as: (1) where the time provably goes, (2) decode
levers ranked, (3) prefill levers ranked, (4) the DSpark-head question the
user asked about, (5) doors that are measured-shut (do not reopen without new
evidence), (6) small code-level observations made while reading.

---

## 1. Where the time goes (all figures from the repo's own measurements)

### Decode (one block ≈ 600–860 ms, yields ~2.44 accepted + 1 bonus tokens)

Per verify layer, spine-observed wall **12.6 ms**; mean worker-side layer max
**10.28 ms**; **~2.3 ms/layer spine-side issue/merge overhead, unattributed**
(2026-08-07 runs.txt, "STRAGGLER ATTRIBUTION CORRECTED"). Per R9700 request
(7.96 ms wall, 8.5 experts, 1.64 page-ins):

| leg | time | share |
|---|---|---|
| `ns_read` (cold-read chain) | 4.55 ms | 57% |
| `ns_h2d` (staging→VRAM) | 1.66 ms | 21% |
| `ns_submit` | 0.96 ms | 12% |
| rest (prep/lookup/compute) | ~0.8 ms | 10% |

- The isolated 1-expert graph is **87 µs** — *compute was never the wall*.
- The drive is **76–90% idle during decode**; reads are bursty and
  request-scoped-serial. Idle bandwidth exists; the blocker is knowing *what*
  to read (see §4).
- Draft (3 NextN layers via the CPU worker, OMP=8): ~25–50 ms/block, ~5–9% of
  block time. Compute-bound (81% `ns_submit` = raw matmul), 4.3% page-ins.
- Decode is a **strictly serial alternation**: spine layer-L attention+router
  (GPU) → blocking dispatch round-trip → spine layer L+1. Worker idles during
  spine compute; spine GPU idles during worker service. Neither overlaps the
  other — there is no pipeline to deepen, only latency to remove.

### Prefill (one sweep at 493-tok prompt ≈ 30 s; 26.15 t/s at 2084 tok)

- Read leg: ~67.5 GiB page-ins per 493-tok prompt; every page read exactly
  once (zero amplification). 2026's SN750 is the slow shard (3.94 GB/s gen3
  x4 link shared by both workers; 2.5–3 GB/s to RAM measured in July).
- **Wire leg: `issue` = 8.2 s of a 26.4 s prefill dispatch.** ~820 MB of f32
  activations cross the 1 GbE per sweep *before gather*; the spine-side gather
  (68% row need) removes ~263 MB of that. The spine's `issue_requests()`
  sends **synchronously, one worker at a time** — `send()` of a ~10–40 MB
  frame blocks once the 4 MiB `wmem` fills, so the spine thread sits on the
  wire while workers could already be reading.
- Codex's corrected ceiling: **~42 t/s hard ceiling on the R9700 leg at
  6.2 GB/s**; 80–100 t/s needs hardware (more/faster storage or +35–45 GiB
  host RAM). Software headroom is ~2–2.5×, not 5×.

---

## 2. Decode levers, ranked

Ordered by (expected gain × confidence) ÷ effort. Nothing here reopens a
measured-shut door (§5).

### D1. Async issue on the spine + remote-first send order — *new, cheap, attacks the unexplained 2.3 ms/layer*

**Observation.** `pipe-expert-dispatcher.cpp:issue_requests()` loops over
workers calling `pipe_send_frame()` — a **blocking** `send()` loop
(`pipe-transport.cpp:send_data`). At verify width the payload is only
~100–200 KB/worker, so this is sub-ms — but it is on the critical path *every
layer*, and it serializes worker start times: the last-issued worker starts
its NVMe reads after the earlier sends complete. The ledger's unexplained
"~2.3 ms/layer spine-side issue/merge overhead" (18% of the layer clock) has
never been decomposed past "not transport" (the TCP-vs-WireGuard arm was
neutral, which clears the *network*, not the *send path*).

**Proposal.**
1. Per-worker writer thread with a per-socket FIFO queue. The dispatcher
   enqueues and returns; responses are awaited exactly as today. FIFO per
   socket preserves every wire-order invariant the deferred-fold logic relies
   on (N-1 deferred sent before N on the same socket). No protocol change, no
   worker change, no PIPE_VERSION bump.
2. Cheap interim arm: **issue remote workers before loopback** (reorder the
   `issue_requests` loop). ~15 lines, same determinism.

**Expected:** removes most of the 2.3 ms/layer if issue is indeed the bulk of
it → up to ~15% decode. **Gate:** decompose first — one run with
`WP_DISPATCH_REQ_LOG` + `WP_DISPATCH_STATS` already gives `ns_before_await`
vs `ns_blocked` per request; if issue isn't the 2.3 ms, don't build the
threads.

### D2. Split-frame dispatch: assignments first, activations second — *new, medium effort, stacks with D1*

**Observation.** A worker cannot begin reading pages until the **entire**
frame has arrived: `pipe_recv_frame` blocks until the payload (assignments +
the full f32 activation blob) is complete. But `ensure_batch()` only needs
the **assignment list** — page-ins don't need activations; only `compute`
does. On the 1 GbE link, the activation leg is ~1.2–1.8 ms at verify width
and ~90–170 ms/layer at prefill widths. The remote workers are the straggler
in ~50% of layers (1070: 26.4%, 480: 23.1%).

**Proposal.** PIPE_VERSION 6: send `{layer, n_tokens, assignments}` as frame
1, `activations` as frame 2. Worker: `ensure_batch()` (reads start) on frame
1; `prepare_io()` on frame 2; compute joins both. Reads then overlap the wire
instead of following it. Determinism untouched (same bytes, same order).

**Expected:** decode ~+3–6% (remote-start delay × straggler frequency); the
prefill payoff is larger (see P1).

### D3. Re-adjudicate WP_HIP_GRAPHS + WP_EXPERT_GRAPH_CACHE + GATHER_MIN_TOKENS≥8 in a quiet window — *built, unmeasured*

The D2 graph cache is mechanically proven (92.8% hit, byte-identical twice),
but **it only engages on the dense path** (`!use_gather`), and the config of
record runs `GATHER_MIN_TOKENS=1` (gather always). The `gc16` arm paired the
cache with GM16 (dense at verify widths) and posted the best worker walls of
its window (pg0 2.02 ms); then round 7's control beat the combo in a drifting
window. This is the highest-value *already-built* decode configuration and it
has never had a clean measurement. One quiet-window A/B/A/B, canonical
trajectory required. If it holds: promote `GCACHE=1 + GATHER_MIN_TOKENS=8` to
the config of record (verify goes dense+cached; prefill keeps gather).

### D4. True async H2D on a dedicated copy stream (CUDA/HIP) — *the consult lever, only half-built*

`WP_EXPERT_ASYNC_H2D` measured **negative** because the copy issues on the
*compute* stream — same-stream ordering still gates the graph on the last
stripe, so it keeps the serialization and adds event overhead (ledger, round
5, byte-identity confirmed twice so the fencing design is sound). The actual
lever is a **second stream**: issue the staging→slot copy on a copy stream,
record an event, and have the compute stream wait on the event. That hides
the 1.66 ms `ns_h2d` (21% of the R9700 request) under the tail of the reads.
On the 1070 the drain is 4.4–5.6 ms/pg+ request (staging is pageable there —
and note the discovery that it has been `pinned` all along via an empty-var
harness quirk, so pageability is *not* its limiter; the drain path is).
Vulkan stays sync (async kills the 480 worker; guard already in place).

**Note on the bigger version of this idea (P2P / SAM / ReBAR, NVMe→VRAM
direct):** the machinery exists (`wp-file-io-p2p.cpp`, io_uring + DMA-BUF,
HIP-only) but its July measurement was **1.7× *slower* per read than the HOST
path at identical concurrency** (3.29 vs 4.39 GB/s), and the window-cache fix
was retracted as net-harmful. 2026 has no ReBAR at all. Treat "direct
NVMe→VRAM" as *unproven*, not as the roadmap default; D4's copy-stream
version is the low-risk 80% of it.

### D5. Eviction-policy sweep, offline, zero GPU — *headroom is proven and bounded*

Belady bounds the recoverable miss count at **34–38% of LRU's page-ins**; the
current use-count policy captured ~10% of that. The reference stream is
policy-independent and already capturable (`WP_REF_LOG`), and `sim-evict.py`
predicted the measured LFU delta to 0.11%. Untried candidates that fit the
workload's known shape:

- **S3-FIFO / WTinyLFU-style admission**: the prefill sweep is a *scan* — every
  page read once, never re-referenced — running through the same pool decode
  lives in. Scan resistance is exactly what these policies add. (2Q tried and
  failed, but 2Q's queue sizing is the fragile part; S3-FIFO's small-probation
  FIFO is a different mechanism.)
- **Sweep-boundary handling**: at the prefill→decode transition, the pool
  holds the *tail of the last sweep*, i.e. the pages least likely to be needed
  first by decode. A one-line "admit prefill pages at lowest rank" rule (they
  already get `uses = evict_age_ + 1`; try `lease_until = 0` + prefetch-band
  tick for n_tokens>1 page-ins) is simulatable offline before any GPU run.

**Expected:** each 10% of page-ins recovered ≈ 0.45 ms/layer ≈ 3–4% decode.
Honest ceiling ~10% decode. Cheap to explore; do it before any new residency
hardware conversation.

### D6. RX 480 tail suppression — *the 480 paces 23% of layers with 14–17 ms severe tails*

1. **VKFIX has never once been armed.** `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM`
   was dead code from the day it was written (the `VKSPLIT` `:-` bug, fixed
   but never re-measured). One interleaved arm, verified via
   `/proc/<pid>/environ`. Free.
2. The **6× in-worker submit divergence** (live ~1510 µs vs 190 µs isolated)
   has a named, untested suspect: H2D serializing against compute on a single
   Vulkan queue. If the driver exposes a transfer queue, routing
   `tensor_set` there is the Vulkan analogue of D4.
3. **Weighted static assign**: `choose_worker`'s splitmix picks uniformly
   between the 1070 and the 480 for experts 0..84. A deterministic bias (hash
   modulo a weight vector, still a pure function of `(layer, expert)` — the
   reproducibility property is preserved) toward the 1070 shifts severe-tail
   layers to the card that drains at 2.9 GB/s instead of the one with the
   14–17 ms pathology. Same shard, same drive; no re-sharding. One arm.

### D7. Draft block cost & the CPU-worker's nondeterminism — *two birds*

The draft is ~25–50 ms/block on the CPU worker (OMP=8, compute-bound).
During the draft, the 2026 GPU workers are **idle**. `ES_2026_DSPARK` (the
mirror shard) already exists on 2026's NVMe; the `DSPARK_SPLIT` harness path
already knows how to place blk.43-45 on two CPU workers. An arm that serves
draft experts 0..84 from the 1070/480 (which hold those expert *ranges*
already for layers 0–42 — the pages are disjoint per-layer, but the pools and
drive are shared) would price GPU drafting directly.

**Second reason to do this:** the residual cross-run nondeterminism (3+
sightings; first-arm divergence pattern twice; KG suspect: "CPU DSpark worker
threaded reduction order, 2026-08-03") lives in the CPU worker. Every sag /
divergence window costs measurement days. Moving the draft onto deterministic
GPU backends is both a speed arm *and* removes the leading nondeterminism
suspect. If the arm is slow, still consider running the CPU worker with a
fixed-thread-order reduction for measurement hygiene.

### D8. turbo4 — keep expectations scoped (expanded after the "does turbo4
help t/s?" question)

**At the benchmark operating point (CTX=8192): ~nothing.** Decode is ~80%
dispatch wait; spine attention is a slice of the 2.3 ms/layer overhead. Even a
perfect KV-attention kernel is ~0.2–0.4 ms of a 12.6 ms layer wall → low
single digits end-to-end. Prefill is drive-bound: spine attention is ~1–2 s
of a ~30 s sweep, and rung 1's −11.4% FA prefill shape is ~150 ms of it.
The workers never see turbo4 at all — KV lives only on the spine; the worker
cards run expert FFNs.

**The 1.96×/1.49× figures are corrupted-state flattery** — measured SPEC-off
while turbo4 produced degenerate output (acceptance 0.225); degenerate text
routes to the same few experts, the expert cache stops missing, and the
paging system falls out of the measurement. The honest baseline is the 08-04
finding: f16 faster (6.52 vs 5.29) and cleaner (0.953 vs 0.897). DS4 is a
structurally poor fit for KV quantization anyway (csa is a /4 compressor
output, hca /128, and the lightning-indexer cache refuses quantization
outright).

**Where turbo4 genuinely matters — the context goal, not the throughput
campaign.** 1M f16 KV (~6.6 GiB) does not fit on the 16 GB 6900XT beside the
7.9 GB dense model; turbo4's ~1.8–2.8 GiB is what makes 1M possible at all.
And at 256K–1M the spine's KV-read traffic grows linearly with context until
attention becomes a first-order decode cost — turbo4's ~4× smaller KV is the
lever *there*. The crossover is somewhere in the hundreds of K, but it only
becomes reachable after the KV bytes/token regression (21504 vs 6880 B/tok,
serve-debug §3) is root-caused, since nothing above ~160K fits tonight.

**Sequencing:** run the owed temp-0 byte A/B (cheap, and the corruption needs
fixing for the context goal regardless), bank rung 1 (done), re-run rung 2 in
a cool window — but do not fund turbo4 out of the prefill/decode campaign's
budget. It is not on the path to 40/8; it is on the path to "512K+ serving
works at all."

### D9. Micro (worker): persistent reader threads; spec-queue drain rate

- `ensure_batch()` **spawns `std::thread`s per request** (up to 4) and joins
  them in `complete_batch()`. pthread create+join ≈ 30–60 µs each, paid by
  every request with ≥1 page-in — i.e. most decode requests. A persistent
  reader pool (or `std::async` pool) is a small, safe change worth
  ~0.1–0.2 ms/request. On the 480 (where everything is 6×) it may matter
  more.
- `find_slot` / `select_victim` are O(n_slots) scans per page (2200 slots on
  the R9700). Trivial at decode widths, but it's ~300K compares per prefill
  request per layer — a hash map on `(layer, expert)` is ~30 lines and takes
  `ns_lookup` to ~0. Low priority; include only if `ns_lookup` shows up in
  REQLOG.
- Spine-side `update_residency()` does vector erase+push per assignment
  (O(slots) memmove each): ~100–200 µs/layer at prefill widths. A
  `list`+`unordered_map` LRU kills it. Also inside the D1 measurement.

---

## 3. Prefill levers, ranked

Context: at UBATCH=2048/prompt 493 the union per layer covers ~32–40 of a
worker's ~85–171 experts (RX 480: 1381 references/1352 page-ins per sweep) —
**not** saturated; reads ≈ coverage × shard. Drive util already hits
80–115% during prefill bursts, so *ordering* wins are small; the wins are
*overlap* and *not re-reading*.

### P1. Overlap the wire leg with the read leg — the biggest prefill software lever

`issue` = **8.2 s of 26.4 s** of prefill dispatch, and it is structurally
serial today: the spine blocks in `send()` per worker, and the worker blocks
in `pipe_recv_frame` until the whole frame (incl. ~10–40 MB of f32
activations) lands, *then* starts reading.

- **D1 (async issue)** alone moves the spine's send-blocking off the
  dispatch thread: the spine can run layer L+1's attention while layer L's
  frames are still draining to 2026.
- **D2 (split frame)** lets reads start ~90–170 ms/layer earlier on the
  remote workers — reads overlap the wire, per layer, both machines.
- Combined estimate: most of the 8.2 s issue leg is recoverable against a
  ~30 s sweep at 493 tok → **+20–35% prefill**. This is the only software
  lever of that size left on prefill.

### P2. OFFSET_SORT deserves its long-prompt arm before closure

`WP_EXPERT_OFFSET_SORT` (byte-identical (blob,offset) read ordering) was
tested only at 493 tok, where it posted 13.69 — the low end of the noisy
band, but 493-tok prefill is ~31 page-ins/request with the drive already
queued 11–14 deep; sorting can't show its seek win there. The code2000 arm is
already owed per the ledger. Keep it queued; don't close on the 493 result.

### P3. PREFILL_GATE arm (built, never run)

`WP_EXPERT_SPEC_PREFILL_GATE=1` pauses speculative submission during
prefill-shaped requests (spec LATE runs 84–100% there = pure contention
against a demand stream that misses everything). Supporting evidence:
`specoff` posted the **best prefill of the day (17.15)** — no draft
contention during prefill. One interleaved arm on the config of record:
`PREFILL_GATE=1 PREFETCH_HINT=1 SPEC_PAGEIN=1`. If it reproduces even half of
the specoff delta while keeping decode hints, adopt.

### P4. A store-and-forward relay on 2026 — *halves cross-machine prefill bytes*

Both 2026 workers receive nearly the same activation rows (68.0%/67.9% union;
their mutual overlap is large). The spine sends each copy separately over the
single 1 GbE. A tiny relay process on 2026 (one cross-machine TCP read,
fan-out over loopback) cuts the cross-machine wire leg ~2× for prefill.
Determinism-neutral (same bytes delivered). Moderate build; only worth it if
P1's decomposition shows the wire leg survives D1+D2. Alternative with zero
new processes: spine sends the shared row-set once + per-worker deltas —
protocol work, probably not worth it vs the relay.

### P5. The tail sweep

code2000 (2084 tok) = 2048 + 36: the 36-token remainder pays a near-full
sweep (~40 s vs 32.5 s) because the pool has turned over ~3× during sweep 1.
Nothing cheap fixes the physics, but: (a) size `--ubatch-size` to the
deployment prompt-length distribution so typical prompts don't straddle a
boundary; (b) if 2-sweep prompts are common, a **sweep-2 admission rule**
(D5's prefill-band admission) at least stops the tail sweep from flushing the
pages decode is about to need. Both are policy, not plumbing.

### P6. Honesty line

With reads at the drive ceiling and zero read amplification, **prefill's
software ceiling is ~25–40 t/s** depending on prompt length; Codex's 42 t/s
R9700-leg bound is the right mental model. 80–100 t/s is a hardware
conversation (a second NVMe on 2026 — the two workers share one gen3 x4
link — is the cheapest real prefill upgrade; host RAM for tier coverage is
the second).

---

## 4. The DSpark-head question — what "more potential" actually remains

Your instinct is half-right, and it is worth being precise about which half,
because three variants of this idea have already died measured deaths and a
fourth is untested.

**Dead (do not reopen without new evidence):**
- *Cross-layer router arithmetic* `router_{L+k}(h_L)` (WP_PREDICT_AHEAD):
  per-rank precision is real (94.5% @ k=1) but **incremental precision tops
  at 21%** — router confidence and LRU residency are the same signal, so the
  pages it supplies cleanly are pages the cache already holds. The miss tail
  is exactly what it cannot call. 18 arms, every cell ≤ baseline. Closed for
  mean t/s; kept as free tail insurance (k2m2+preempt, 463 promotes/run at
  0 cost).
- *MTP depth transfer* (draft routers as proxy for target routers): 2/40
  softmax layers cleared the 0.60 gate. Closed.
- *The 2026-07 DFlash adapter*: 0.14 recall@6. Closed.

**Alive — and it is the one variant with qualitatively more lead time:**

The verify batch's **entire token set is known before verify starts**, and
the DSpark draft's final NextN embeddings for those tokens
(`llama_get_embeddings_nextn`, 16384-wide rows) exist at the same moment.
That means a predictor keyed on draft hidden states can emit expert hints
for **all 43 layers of the verify pass with up to ~500 ms of lead** — versus
1–2 layers (sub-10 ms) for every attempt that died. Lead time, not prediction
quality, was the verdict of all three post-mortems. And the July pollution
failure mode is now guarded by machinery that didn't exist then: host-RAM
landing (`SPEC_HOST`), short predicted leases, the preemptible slice reader,
and the two-band LRU.

**Proposal — learned probes, offline first, zero GPU risk:**

1. *Capture.* Draft-side: per verify block, dump (block_id, token ids, nextn
   embeddings). Target-side: `WP_PREDICT_CAPTURE` already dumps
   (layer, h_L, actual selections) per step. Join on position. ~40 lines,
   modeled on the existing capture; capture arms are explicitly not banking
   reps (0.35 t/s cost).
2. *Fit offline.* Per-layer (or per-layer-group) linear probes
   `W: 16384 → 256` trained on the capture; possibly a small MLP for the
   hard layers. This is a CPU/numpy project.
3. *The day-one falsifier, stated in advance:* score **recall of the
   first-non-resident set** (the actual miss stream), at fixed hint volume
   (e.g. M=8–16/layer), with a simulated pool — not rank-1 precision, not
   union recall (the 08-07 metric conflation). The bar to beat is the
   router-k=2 first-non-resident baseline: **28.3% @ depth p50=4**. If a
   learned probe can't clearly beat that *on the miss stream*, the whole
   direction closes for good, cheaply, with no fleet time.
4. *If it clears:* wire it as PREDICTED hints at draft end (the queue,
   dedup, host landing, and preemption all exist), volume-gated, prefill
   gated off. Target: convert a slice of the 4.55 ms/layer read chain into
   idle-drive reads. Even 25% conversion ≈ 1.1 ms/layer ≈ **+8–10% decode**;
   the ceiling is bounded by the residency correlation that capped every
   predecessor, so treat >15% decode as upside, not plan.

**Cheaper cousin worth one offline afternoon:** a pure
`(token-id history → layer, expert-set)` n-gram/Markov table, no hidden
states — code corpora are repetitive and consecutive-token expert overlap is
0.33–0.42 (17× chance). The capture for it is token ids + REF_LOG selections.
If it can't beat 28.3% on the miss stream either, that further sharpens the
"prediction is closed" verdict at zero cost.

---

## 5. Measured-shut doors (from the ledgers; do not reopen without new evidence)

- Quantization of the routed experts (already MXFP4 QAT; no representation
  lever exists).
- Re-sharding the expert split (standing constraint).
- WP_DEFER_K: defers by count, not coldness; −10 acceptance points *and*
  −0.26 t/s at K=4. Settled-no.
- WP_DISPATCH_HARVEST: +0.8%, inside noise. The mechanism works; the workers
  were already overlapped.
- TCP-vs-WireGuard: neutral. Transport is not the 2.3 ms.
- READ_STRIPES/READ_WORKERS beyond current: `ns_read` flat at 8 threads/6
  stripes — the demand read path is at its floor.
- KEEPALIVE on the R9700: neutral. (It is a Vulkan-idle-recovery fix; the
  6900XT/R9700 don't have the pathology.)
- Same-stream async H2D: negative (see D4 for the unfixed version).
- Router-arithmetic prediction for mean t/s: closed (§4); remains as free
  tail insurance only.
- `GATHER_MIN_TOKENS=2` alone: neutral (d1gm2 = keep).
- Widening SPEC_NMAX beyond 7: byte-identical — the conf gate binds first.
  The conf knee is workload-dependent (0.6 prose / 0.4 code); retune per
  deployment corpus, not per run.

Also confirmed while reading (so nobody spends an arm on it): **the DSpark
draft is already greedy.** `common/speculative.cpp:1357` reads
`cur_p->data[0].id` — the post-top-k argmax — and discards the sampled
`cur_p->selected`. There is no acceptance win hiding in the draft sampler.
Acceptance gains can only come from drafter quality/config, and the knee was
swept.

---

## 6. Smaller observations from the code read

1. **Mask-token hint noise.** The per-ubatch hint (`llama-context.cpp:2301`)
   fires during the *draft* decode too, where the batch is
   `[id_last, mask×k]` — so `tid2eid` rows for `mask_token_id` get hinted for
   layers 0–2. The dispatcher's `(layer, provenance)` dedup eats the repeats
   after the first block, so the cost is one garbage frame-set per session —
   but filtering `token == mask_token_id` at the hint site is three lines and
   removes a class of confusion from the hint logs.
2. **The KV-bytes/token regression (21504 vs 6880 B/tok)** in
   `2026-08-07-serve-debug.md` §3 blocks every long-context serve config and
   is orthogonal to all of the above — worth its owner before the turbo4
   context story can matter.
3. **The udev rule for the RX 480's runtime PM is still not installed**
   (staged at `/var/tmp/99-rx480-no-runpm.rules` on 2026, awaiting sudo).
   One reboot silently restores the 7.5 GB GTT-eviction pathology and the
   20% memory-pressure windows. One command; do it at the next maintenance
   window.
4. **`WP_EXPERT_SPEC_PREFILL_GATE` and `VKFIX` are both built and both have
   never executed a measured arm.** Two of the cheapest open runs on the
   board (P3, D6.1).
5. **The evening measurement window is instrument-limited** (±4% drift,
   unattributed; Brave convicted for the daytime sag). D3 and the turbo4
   rung-2 verdict both need a quiet/late-night window; until then, treat all
   sub-5% verdicts from the evening of 08-07 as provisional.

---

## 7. Suggested order

| # | action | cost | expected |
|---|---|---|---|
| 1 | D1 measurement: REQLOG join to decompose the 2.3 ms/layer | one run | decides D1/D2 build |
| 2 | P3 + D6.1 (PREFILL_GATE, VKFIX) — two built-never-run arms | two arms | unknown, free |
| 3 | D3 quiet-window A/B/A/B (GCACHE+GM8, HIP graphs) | queued already | up to +5–7% decode |
| 4 | D1 build (async issue + remote-first) | ~1 day | up to +15% decode, +20–35% prefill with D2 |
| 5 | §4 capture + offline probe falsifier | CPU-only | gates the last big decode idea |
| 6 | D4 copy-stream H2D (CUDA first, HIP second) | 1–2 days | ~1.6 ms/layer on paged requests |
| 7 | D5 offline policy sweep (S3-FIFO / admission / sweep boundary) | CPU-only | 0–10% decode |
| 8 | D6.3 weighted assign, D7 GPU-draft arm | one arm each | tail suppression; determinism hygiene |
| 9 | D2 split-frame protocol | ~2 days + rebuild-all | stacks on D1; prefill-critical |

Nothing above touches the standing constraints: no re-sharding, no
quantization, no builds or GPU runs were performed for this analysis.

---

## 8. Realistic ceilings (added on follow-up discussion)

### Decode

- **Realistic software-only landing zone: 6–8 t/s.** Every known lever lands
  well: read chain at stripe floor (~4 ms → ~2 ms), h2d hidden on a copy
  stream, graph cache + GM8 on verify, D1/D2 killing most of the 2.3 ms/layer
  spine overhead → layer wall ~6–7 ms, draft ~25 ms → ~3.4 tokens per ~300 ms
  block.
- **10–12 t/s is physically coherent but gated on the §4 learned predictor**
  beating its pre-registered falsifier. The bandwidth check shows the size of
  the prize: at 73% residency the cold bytes are only ~350 MB per block —
  ~40 ms of drive time at aggregate bandwidth versus the ~500 ms actually
  spent, because the reads are request-scoped serial bursts. Perfect
  prediction (not attainable) caps at ~11–13 t/s at current block yield.
- **15–20 t/s = full residency (hardware) plus a healthier drafter.** Note
  the unpriced question: mean accepted length is 2.44 at conf 0.4 against a
  historical 3.5–5.9 for what may be the same head — if that gap is config
  rather than checkpoint, fixing draft yield is the cheapest ~1.5–2×
  multiplier on the board and stacks multiplicatively with everything above.

### Prefill

- **Physics:** ~67.5 GiB of page-ins per sweep, zero read amplification,
  MXFP4 already minimal. Aggregate drive bandwidth ≈ 9 GB/s (6.2 main +
  ~3 on 2026's shared gen3 x4) → drive-limited sweep ≈ 8–9 s. Today: ~30 s
  at 493 tok, ~80 s at 2084 tok (two sweeps) = ~35–45% of the drive limit.
  The gap is per-layer serialization (wire leg, spine compute, drain tails).
- **Realistic: 35–45 t/s at ≥1500-token prompts** with D1+D2+P2+P3 landed
  (~60–75% of drive limit). Longer prompts amortize better; tail-sweep fixes
  help prompts that straddle a ubatch boundary.
- **80–100 t/s needs ~18–20 GB/s aggregate read bandwidth** — a second NVMe
  on 2026 (the two workers share one gen3 x4 link; cheapest real upgrade),
  likely gen4-class on both ends, *and* all the overlap working. Not a
  software target.

### Asymmetry to internalize

Prefill's ceiling is **bandwidth physics** — the levers are low-risk and
nearly additive, climb toward 40 with confidence. Decode's ceiling is
**latency + prediction quality** — past ~8 t/s it runs through exactly one
unproven idea (learned draft-hidden predictor) plus one unpriced question
(draft yield). Suggested campaign order: bank the decode freebies (D3, D6,
D1) to lock ~5.5–6, run the §4 falsifier offline to learn whether 10+ is
alive, push prefill overlap toward 40 in parallel, then have the hardware
conversation with measured ceilings rather than estimates.

---

## 9. Implementation details & caveats, per lever

The "how" for every recommendation above, including the invariants that have
bitten this codebase before. File paths relative to repo root.

### D1 — async issue + remote-first send order

**Where.** `src/pipeline/pipe-expert-dispatcher.cpp`:
`impl::issue_requests()` (~line 940). `planned_request.payload` is already
fully encoded at plan time (`pipe_encode_expert_dispatch_req` in
`plan_requests`), so the writer threads only move bytes — no encode off-thread,
no lifetime questions beyond the payload vector itself.

**Design.**
- One `std::thread` + `std::deque<std::vector<uint8_t>>` + mutex/cv **per
  worker socket**, created in `impl::impl()` after HELLO, joined in
  `poison()`/destruction. `issue_requests` enqueues `{seq_id, payload}` and
  returns.
- **Per-socket FIFO is the only ordering guarantee needed and it must be
  total.** The deferred-fold path relies on "N-1 deferred frames were SENT
  before N's frames on the same socket" (comment at `collect_pending_deferred`)
  and `await_response` validates `seq_id`. A single writer thread per socket
  preserves both for free. Do NOT pool sockets across threads.
- **`send_prefetch_hints()` must go through the same writer queue.** Hints are
  currently sent from the dispatch thread (`graph_dispatcher::compute` →
  `flush_predicted_hints` / `prefetch_for_tokens`). Two threads calling
  `send()` on one socket can interleave bytes *inside* a frame — that is wire
  corruption, not a reorder. Route hints through the queue like everything
  else. (The reverse hazard — a hint overtaking a request — is harmless:
  hints are seq_id 0 and advisory.)
- **The `in_flight != 0` hint gate needs rethinking.**
  `send_prefetch_hints` declines when `in_flight != 0` so a hint can never be
  interleaved ahead of an uncollected deferred partial's response matching.
  With async sends, increment a *logical* in-flight at enqueue, but the hint
  gate needs *wire* quiescence: expose `wire_idle()` = (all writer queues
  empty && in_flight == 0) and gate on that. Otherwise hints silently stop
  being sent during every dispatch window (they'd look offered-but-skipped in
  `n_skipped_in_flight` — watch that counter in the first arm).
- **Error propagation.** `pipe_send_frame` failure currently throws
  synchronously from `issue_requests`. Async: the writer records the failure
  into an atomic + stored message; checked at the next enqueue and at
  `await_response` entry. On any failure call the existing `poison()` —
  sockets are already torn down there. Never let a writer thread throw
  uncaught (std::terminate).
- **Backpressure:** none needed in practice (one layer of requests ≈ 3 frames
  ≤ ~60 MB worst-case prefill), but assert-cap the queue at, say, 8 frames to
  fail loudly instead of growing without bound if a worker wedges.
- Recv side (`pipe_recv_frame` in `await_response`/`harvest_partials`) stays
  on the dispatch thread. TCP is full-duplex; no lock needed between writer
  thread and recv thread on the same `pipe_socket_t` — but audit
  `pipe_socket_t::impl` for shared mutable state first (today there is none
  beyond the fd).

**Cheaper interim arm (do this first, same day):** reorder the
`issue_requests` loop to issue remote workers before loopback. The worker
order is fixed at connect time (`workers[]`); just stable-partition the
requests vector by `target.machine != local`. No threads, no queue — if this
alone moves `ns_wait`, it prices the serialization before the threaded build.

**Measurement.** `WP_DISPATCH_REQ_LOG` (spine) + `WP_REQ_LOG` (workers),
joined per request: watch `ns_before_await` (issue→await-start) collapse and
the workers' `ns_wait`-equivalent start earlier. Canonical trajectory gate as
usual (wire bytes are unchanged, so acceptance must not move).

### D2 — split-frame dispatch (assignments first, activations second)

**Where.** Protocol: `src/pipeline/pipe-protocol.{h,cpp}` (add
`PIPE_EXPERT_DISPATCH_BEGIN` / `PIPE_EXPERT_DISPATCH_ACTS`, PIPE_VERSION 5→6).
Spine: `plan_requests`/`issue_requests` split the payload. Worker:
`serve_connection` frame loop in `tools/wp-expert-worker/wp-expert-worker.cpp`.

**Worker-side sequencing.**
- On BEGIN: validate layer/experts exactly as `Worker::dispatch` does today,
  then immediately `pool_.ensure_batch(pages, ...)` — **this starts the NVMe
  reads** — and stash the returned `Batch` in a `std::optional` member. Also
  raise `pool_.demand_serving(true)` here (the current RAII gate wraps
  `dispatch()`; with the split, reads start at BEGIN, so the preemptible
  landing gate must too).
- On ACTS: `prepare_io`, the chunked compute loop, `read_result`, send the
  partial — the existing code path, unchanged, reading the stashed Batch.
- **Frames that can legally arrive between BEGIN and ACTS on the same
  connection:** `PIPE_PING` and `PIPE_EXPERT_PREFETCH_HINT`. Both are handled
  before the dispatch branch today; keep it that way. Anything else while a
  BEGIN is outstanding = protocol error.
- **The stashed Batch pins its slots while waiting for ACTS.** Bounded —
  ACTS follows within a wire-leg — but the connection-close path must call
  `pool_.abandon_batch()` on a pending BEGIN, or the worker leaks pins and
  eventually deadlocks `select_victim`. `abandon_batch` exists; wire it into
  the `serve_connection` exit path.
- The two frames share the request's `seq_id`; echo it in both and check on
  receipt. Response frame unchanged (one PARTIAL per pair), so
  `await_response`'s seq_id accounting is undisturbed.

**Caveats.**
- **PIPE_VERSION bump = rebuild spine AND all four workers together** or HELLO
  refuses — this has repeatedly read as a worker crash. The harness's build
  check (`grep -c WP_EXPERT_READ_STRIPES ...` pattern) should gain a protocol
  check for this.
- `spec_prefill_gate_active_` is set from `request.n_tokens` at `dispatch()`
  entry — n_tokens arrives in BEGIN, so the gate timing is unchanged.
- The determinism properties live in compute order (assignment index), which
  is fully known from BEGIN. Splitting changes *when* reads start, never
  *what* is computed or in what order.
- Keep the monolithic frame as a fallback (`WP_SPLIT_FRAME=0`) for one arm
  cycle, then delete once D1+D2 are both banked — two wire formats is a
  permanent test-matrix tax otherwise.

**Measurement.** Worker `WP_REQ_LOG` `ns_wait` (request-scope read wait)
should fall by ~the wire leg on remote workers (~1.2–1.8 ms decode,
~90–170 ms/layer prefill); spine `ns_blocked` should fall by the same.
Interleave against monolithic-frame control, canonical trajectory required.

### D3 — GCACHE=1 + GATHER_MIN_TOKENS=8 promotion

Config-only, but the caveats are the point:

- The graph cache engages only when `!use_gather`
  (`wp-expert-worker.cpp:4442` area), so `GCACHE=1` without
  `GATHER_MIN_TOKENS>=8` does **nothing at verify widths** — gather is default
  ON at `n_tokens >= 1`. The arm is the *pair*, plus `COALESCE=1` (cache
  requires the coalesced-params path).
- Dense-at-verify costs ~2.3× expert FLOPs (verify routing density) — the
  bet is that ~0.96 ms/req of submit + graph-build savings beats that at
  width ≤ 7. The 87 µs isolated-graph number says compute is cheap; that is
  why this is plausible, not proven.
- Cache entries pin their gallocr VRAM (~2–4 MB × `WP_EXPERT_GRAPH_CACHE_MAX`,
  default 16). On the 550-slot 8 GB cards that's fine; on the R9700 with 2200
  slots (28 GB spoken for) confirm free VRAM before arming.
- Cached graphs are invalidated by io/params buffer growth (`io_gen_` /
  `params_gen_` bumps). With `WP_IO_PREALLOC_TOKENS` set to UBATCH there
  should be zero growth in steady state — if `gcache_miss` climbs
  run-over-run instead of plateauing, a buffer is growing mid-serve; check
  the `wp io-buffer grow` lines.
- **Run it in a quiet/late-night window.** The 08-07 evening window drifted
  ±4% on identical canonical work — the same order as the effect under test.
  Gate: canonical trajectory (146/155, acc 0.94194 on code700) AND ≥4
  interleaved reps, per-worker REQLOG walls (pg0/pg+), not tok/s.

### D4 — dedicated-copy-stream H2D

**The gap in the current build.** `WP_EXPERT_ASYNC_H2D` issues
`ggml_backend_tensor_set_async` on the backend's *compute* stream
(`StagingPool`, `wp-expert-worker.cpp:1249+`). Same-stream ordering means the
copy still serializes against the graph — that is why round 5 measured
neutral-to-negative with byte-identity intact. The event fencing
(`mark_in_flight` / `borrow()` waits) is correct and reusable as-is.

**What the real version needs** (ggml has no public multi-stream API — this
is a fork-local backend extension, same spirit as `WP_HIP_GRAPHS`):
- CUDA backend (`ggml-cuda`): add an env-gated dedicated copy stream per
  device; route `tensor_set_async` for weight-sized copies there. Record an
  event on the copy stream after each copy; **the compute stream must
  `cudaStreamWaitEvent` before the next graph launch that reads the slot**.
  Simplest correct placement: one "latest copy event" per device, waited at
  the top of `ggml_backend_cuda_graph_compute` when pending. Coarser than
  per-tensor edges but trivially right, and copies are always followed by
  exactly one graph compute in this worker.
- HIP: same shape (`hipStream`/`hipEvent`). The R9700 is the highest-value
target (0.6 ms drains already; the win is removing even that from the
  request path).
- **Vulkan stays on the sync path, gated by runtime backend *name*** —
  `ggml_backend_name()` contains "Vulkan" — never a `GGML_USE_*` macro. That
  bug class has recurred six times here, and Vulkan advertising events while
  dying under async is exactly the 2026-08-07 crash.
- The staging-reuse fence already handles buffer recycling across streams
  (`borrow()` syncs the recorded event). Keep the runtime-disarm path
  (event creation fails → one full sync + permanent fallback).
- Determinism: copy scheduling never changes values, and the compute-stream
  wait makes ordering total. Byte-identity gate applies anyway (canonical
  trajectory), because this codebase's history says so.
- **1070 note:** its `staging_kind=pinned` turned out to be an empty-env-var
  accident on the VKENV launch path, and pinned was *neutral* — so its
  4.4–5.6 ms drain is not pageability. Before building the copy stream for
  CUDA, instrument *where* the 1070's drain time actually goes (one
  `WP_REQ_LOG` read of `ns_h2d` vs `bytes_h2d` already gives GB/s per
  request; if it's ~3 GB/s with pinned buffers, that's a PCIe/pathology
  question, not an async question).

### D5 — offline eviction sweep

- `docs/dev/sim-evict.py` already loads `WP_REF_LOG` streams and runs
  LRU/FIFO/LFU/2Q/Belady. Add **S3-FIFO** (≈10% probation FIFO + main FIFO
  with re-reference bits) and a **count-min-sketch doorkeeper** (WTinyLFU
  admission). Also add a *prefill-aware* variant once the stream can tell
  phases (next bullet).
- **`WP_REF_LOG` currently records only (layer, experts)** — no n_tokens, so
  prefill and decode requests are indistinguishable in the stream. Append a
  trailing column (the format is positional-from-the-left; the REQLOG
  precedent is trailing additions never move existing columns). Without this,
  the sweep-boundary policies (admit prefill pages into the prefetch band)
  cannot be simulated.
- The worker-side change for a sim winner: one stamp site —
  `drain_one_read`'s `slot.uses = evict_age_ + 1` (and the host-hit restore's
  twin). Plumbing `request.n_tokens > 1` down to `ensure_batch` is a new
  parameter, three call sites.
- Caveat: the sim models *demand only* — no leases, no spec band, no host
  tier. Treat its ranking as a filter: any winner still gets a live
  interleaved A/B, and the live metrics are per-worker `n_pagein` and
  dispatch wait, never tok/s.

### D6 — RX 480 items

1. **VKFIX arm.** The harness fix is in (`VKSPLIT=${VKSPLIT-...}`), so
   `VKSPLIT= VKFIX=1` now reaches the `elif`. **Verify the flag actually
   arrives** before reading any number: `p=$(pgrep -f "device Vulkan0"); tr
   '\0' '\n' < /proc/$p/environ | grep GGML_VK` — this exact check is in the
   handoff because the flag had never once been set in any prior run.
2. **Vulkan queue investigation:** read-only start — check whether the
   ggml-vulkan backend exposes a separate transfer queue on Polaris and
   whether `tensor_set` and compute share one queue. The 6× submit divergence
   (1510 µs live vs 190 µs isolated) with H2D-on-one-queue as the named
   suspect. No code until the queue topology is confirmed from source
   (`ggml/src/ggml-vulkan/`) — do not async anything on this card (two dead
   workers on 08-07).
3. **Weighted static assign.** `choose_worker`'s static path
   (`pipe-expert-dispatcher.cpp:567` area): replace `h % candidates.size()`
   with a weighted pick, e.g. candidates repeated per weight
   (`{1070, 1070, 480}` → `h % 3`). It stays a pure function of
   `(layer, expert)`, so bitwise reproducibility is preserved, and because
   `send_prefetch_hints` calls the *same* `choose_worker` with the same
   latch, hints and dispatches can never disagree — that invariant was built
   deliberately; do not introduce a second assignment rule. Knob:
   `WP_DISPATCH_BIAS_1070=2` style; default 1:1 = current behavior. Metric:
   480's severe-tail incidence (21% of its layers at 14–17 ms) and its share
   of slowest-worker layers (23.1%), from the REQLOG 3-way join.

### D7 — draft on GPU / draft determinism

- The plumbing exists: `DSPARK_SPLIT=cpu` in the harness launches a
  mirror-shard worker (`ES_2026_DSPARK` on 2026's NVMe, port 8805). The new
  arm is a **second worker process on the 1070** (or 480) serving blk.43-45
  experts 0..84 with ~256 slots (3.18 GiB — the whole draft set fits, so
  ~zero page-ins after warmup).
- **VRAM math first:** 1070 has 8 GB; trunk 550 slots ≈ 7.0 GiB + staging +
  compute. Adding a 3.2 GiB draft worker means cutting trunk slots to
  ~295–300 on that card — and slots have measured ~+10.9% per 100. The arm
  prices "draft latency ↓ vs trunk residency ↓" directly; expect the
  tradeoff to be close, which is why it's an arm and not a plan.
- **Connection topology:** one worker process = one connection = serial
  `accept()`. The draft (ctx_dft) and target (ctx_tgt) each open their own
  dispatcher connections; a dedicated draft worker process keeps that clean.
  Do not try to multiplex two dispatchers onto one worker connection.
- Determinism side-quest (if the GPU arm loses): the CPU DSpark worker's
  threaded reduction order is the leading suspect for the residual
  cross-run divergence (3 sightings). Before touching ggml-cpu, reproduce
  with the draft worker at a **fixed thread count on an idle box** and diff
  partial sums across two runs — if they diverge, the fix belongs in
  ggml-cpu chunking, not here.

### D9 — worker micro items

- **Persistent reader pool:** replace the per-batch `std::thread` spawn in
  `ensure_batch` with N persistent readers fed by a job queue; completion
  still counted per-Batch via the existing `received_` / cv. **Preserve the
  borrow invariant that the stripe-parallel deadlock fix rests on:**
  concurrent readers ≤ staging buffers, and a thread blocked in `borrow()`
  must hold no lease itself. Re-read commit `c9c0c801b` (the sp1/sp1r
  deadlock) before touching this file — the invariant is documented in the
  PageShared comment.
- **`find_slot` hash map:** `unordered_map<uint64_t /*layer<<32|expert*/,
  size_t>` maintained at every `slot.valid` transition — there are exactly
  four mutation sites (`drain_one_read` publish, host-hit restore,
  `ensure_batch` victim clear, `retire_spec_batch` stamping); all run on the
  dispatch thread, so no locking. `select_victim` keeps its scan (it needs a
  min-rank victim, not a key lookup).
- **`update_residency` (spine):** `std::list` + `unordered_map<key, iterator>`
  — the classic LRU pair; removes an O(slots) memmove per assignment.

### P1 — (D1+D2 applied to prefill)

No extra machinery, but the *order* matters at prefill: gather-compaction in
`plan_requests` happens before encoding, and at 2048-token ubatches the
compaction itself (130 weight vectors × 2048 floats copied per worker per
layer) is non-trivial CPU on the dispatch thread. With async issue the
*encode+copy* still runs on the dispatch thread in `plan_requests` — if the
REQLOG join shows `pack` growing, the next step is moving
encode-per-worker into the per-worker writer threads too (the assignments
are computed once; only the per-worker wire image moves). Do not do that
preemptively.

### P2 — OFFSET_SORT long-prompt arm

`OFFSET_SORT=1 PREFILL_GATE=1 PROMPT_FILE=docs/dev/code2000.txt`, interleaved
with control, ≥3 reps. Metrics: prefill t/s, per-worker `ns_wait`, and the
drive profile (`/var/tmp/io-sampler.sh` on 2026 — **start it before the
chain**; the 08-07 sag window cost a day because it wasn't running). Byte
identity is by construction (slots pre-assigned, compute order unchanged), so
a trajectory check is still required but should be a formality.

### P3 — PREFILL_GATE arm

`PREFILL_GATE=1 PREFETCH_HINT=1 SPEC_PAGEIN=1 HINTLOG=1`, interleaved with
the same minus PREFILL_GATE. Confirm from the hint log that (a) prefill-phase
spec submissions drop to ~0 (`spec_pageins` stops growing during the prefill
window), (b) decode-phase hints still land (`used`/`late` per
`analyze-hint-log.py`), (c) acceptance stays exactly canonical. Note the
gate's harvest-continues semantics: in-flight reads still complete during
prefill, only *submission* pauses — so a decode begun mid-prefill-read
inherits at most one in-flight page, bounded by `SPEC_CHUNK`.

### P4 — 2026 relay

- **Measure the overlap first:** two `WP_DISPATCH_UNION=1` runs (one per
  2026 worker is enough — it logs per layer) give each worker's needed-row
  fraction; the *joint* union is what the relay saves. If the two 68%s
  overlap heavily (expected — both are driven by the same routing), the
  cross-link payload drops from ~2×0.68 to ~0.85 of full per layer.
- **Design that preserves response semantics:** one spine↔relay connection
  carrying `[worker_tag][frame]`; the relay splices each tag to the worker
  over loopback, and tags frames on the return path. The dispatcher sees two
  *logical* sockets over one TCP flow — per-tag FIFO preserves every
  ordering invariant (including deferred N-1-before-N, which is per-socket).
  Head-of-line blocking between workers on the shared flow is acceptable:
  the 1 GbE serializes them anyway, and one flow avoids TCP fairness churn.
- The relay itself is ~150 lines of poll-loop C++/Python-is-too-slow-here.
  Put it under `tools/` with the other wp utilities; it must not be a
  process whose death hangs the spine — spine side treats relay-close as
  worker-death (existing poison path).
- Caveat: this changes the wire topology the dispatcher's per-worker speed
  estimator and residency LRU think they see; both key off `workers[]`
  entries, so keep the logical-worker abstraction exact.

### P5 — tail sweep

- Immediate: choose `UBATCH` ≥ the deployment's p95 prompt length when VRAM
  allows (io prealloc scales: `WP_IO_PREALLOC_TOKENS` must move WITH it —
  the 1 MiB-floor lesson, and the io-buffer growth stalls at larger
  ubatch are documented in the warmup comment block).
- Policy half (cheap): admit second-sweep page-ins at the lowest eviction
  rank (prefill band) so the tail sweep doesn't flush what decode needs
  next — simulate first (D5, needs the REF_LOG n_tokens column).

### §4 — learned DSpark-embedding predictor

**Capture (firmware first, ~40 lines).**
- Draft side, `common/speculative.cpp` right after the draft decode
  completes (same place the post-draft hint fires): fwrite the block's
  `llama_get_embeddings_nextn(ctx_dft)` rows (`n_block × 16384` f32 ≈ 459 KB
  per block at width 7), the drafted token ids, and a monotonically
  increasing block id. ~48 MB per 256-token run — fine (cap2 already wrote
  236 MB).
- Target side: `WP_PREDICT_CAPTURE` already writes (layer, h_L, selections)
  per dispatch. **Wrinkle: the custom op never sees token ids** —
  `graph_dispatcher::compute` receives activations only. If you want
  token-keyed joins (needed for the n-gram cousin), plumb `ubatch.token`
  from `llama-context.cpp:2301` into the same capture stream with a record
  marker; the per-layer records already carry (layer, n_tokens) so position
  alignment is recoverable.
- Never bank a capture arm: measured −0.35 t/s from the synchronous fwrites.

**Offline phase (zero fleet time).**
- Per-layer probes `16384 → 256` (softmax regression; shared-trunk MLP only
  if linear clearly underfits). Data budget is the risk: one 256-token run
  gives ~105 blocks × 7 tokens ≈ 735 samples/layer for a 256-way multi-label
  target — thin. Plan several capture runs or a longer server generation
  session (the SERVE mode now works) before concluding anything.
- **Pre-registered falsifier:** recall of the *first-non-resident* expert set
  (the miss stream), evaluated against a simulated pool at 550/2200 slots
  fed by the same run's reference stream, at M ∈ {8, 16} hints/layer. Beat
  **28.3%** (the router-k=2 first-non-resident baseline measured 08-07)
  clearly, or close the direction. Do not report rank-1 precision or union
  recall as the headline — that conflation already produced one retracted
  verdict on 08-07.
- Also evaluate the pure token-id n-gram variant on the same miss-stream
  metric; it needs no hidden states and no training.

**Integration (only if the falsifier passes).**
- Spine-side host copies via the `register_router_oracle` pattern
  (~4 MB/layer); scoring on the existing `predictor_loop` thread
  infrastructure (enqueue at draft end, `flush_predicted_hints` at the next
  layer entry) — do not put a 16384×256 GEMM on the dispatch thread; the
  08-07 sync-GEMM arm priced that at −0.3 t/s.
- Emission: PREDICTED provenance, `SPEC_HOST=1` landing, predicted lease
  (`WP_EXPERT_SPEC_LEASE_PREDICTED`, default 4 — deliberately short), margin
  gate (`WP_PREDICT_CONF` machinery), prefill excluded
  (`spec_prefill_gate` + draft embeddings don't exist during prefill
  anyway). Worker-side dedup and the two-band LRU already do the right thing.
- Volume discipline: the 08-07 volume-ramp lesson — promotes collapse when
  landings are slow, and main's landing queue saturates at M=4. Start at
  M=2/layer and scale only while `promoted/landed` holds.

### Universal caveats for every arm (collected from the trap lists)

- Run from **mad-lab-2026**, never from main (the R9700/DSpark workers
  silently never start otherwise).
- `HOSTVICTIM_2026=0 HOSTVICTIM_MAIN=0` explicitly unless the tier is the
  thing under test — and remember it's **per worker** on 2026 (×2).
- Any `PIPE_VERSION` bump: spine + all four workers rebuilt together; a
  HELLO mismatch reads like a worker crash.
- Interleave arms, never block them; sample load and start the iostat
  sampler before the chain; n=1 tok/s is not a metric — per-worker
  `n_pagein` / wait / REQLOG joins are.
- Determinism gate per arm: acceptance exactly canonical (0.94194 code700 /
  0.84286 prose739) — and capture arms are never banking reps.
- `WP_EXPERT_TIER_VERIFY=1` is diagnostic-only, never in a measured arm.
- Stage hunks, never blanket `git add` — the tree carries 9 dirty files of
  someone else's WIP (`common/speculative.cpp`, `src/llama-context.cpp`,
  the kv-cache set, `src/models/deepseek4.cpp`,
  `tools/server/server-context.cpp`, `fattn-common.cuh`).
- Workers are SIGKILLed at teardown: anything you need to see must be
  fflushed per line (`WP_HINT_LOG` / `WP_PAGEIN_LOG` pattern), not printed
  at exit.
