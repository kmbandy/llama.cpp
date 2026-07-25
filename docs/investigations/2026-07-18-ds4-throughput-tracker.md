# DS4-Flash decode throughput — working tracker

**Goal:** DeepSeek-V4-Flash Q8 (151 GB) decode from **1.73 tok/s → >5 tok/s** on mad-lab-main.

**Status:** open. Started 2026-07-18. Update the STATUS column as items are tried.

---

## 1. Measured baseline (2026-07-18)

| quantity | value | source |
|---|---|---|
| decode | **1.73 tok/s** | `bench-decode.sh`, 256 tok, single stream |
| bytes read per token | **0.97 GB** | `/proc/diskstats` delta over decode window |
| effective read bandwidth | **1.68 GB/s** | 248.92 GB physical / 148.3 s |
| drive clean sequential | **6.2 GB/s** | `nvme_replay.py`, O_DIRECT QD16 |
| **drive utilisation** | **27%** | 1.68 / 6.2 |
| cache hit rate | ~70% | 76.5 MB experts touched vs 22.6 MB read, per layer-token |
| cap/demand | **8.4** | 2167 slots-worth of experts ÷ 258 per-token demand |

**The prize, stated as arithmetic:** at 0.97 GB/token and 6.2 GB/s a token costs 0.156 s = **6.4 tok/s**.
The entire gap between 1.73 and 6.4 is that we run the NVMe at 27% of its measured speed.
**We are I/O QUEUE STARVED, not bandwidth limited.** Every item below is ranked by how much it
raises queue occupancy.

### Expert storage (measured from GGUF data-offset deltas)
- per expert (gate+up+down @ 4.25 bpv) = **12.75 MB** (3 × 4.25 MB sub-pages)
- top-6, one layer = 76.5 MB · all 256 experts one layer = 3.19 GB · all 40 softmax layers = **127.5 GB** (84% of the model)

### Routing statistics (1200 decode tokens, `routing_capture.bin`)
- **222.8 of 256** experts touched per layer → usage is near-uniform
- hottest expert = **6.7%** of accesses (uniform baseline 2.3%) → only 2.9× skew
- weight mass by rank: r1 .297 / r2 .207 / r3 .157 / r4 .129 / r5 .111 / r6 .100 — **flat; no cheap tail**
- pinning coverage, held-out: 54/layer → **0.525**; 128/layer → 0.828 (needs 63.75 GB)

> Load-balancing loss during MoE training deliberately flattens expert usage. The model is
> engineered to defeat exactly the skew a cache depends on. This is why pinning/eviction work
> has poor returns *for us* — see RULED OUT.

---

## 2. Candidates, ranked

Ranked by (effect on queue utilisation × evidence strength) ÷ effort.

| # | item | effort | evidence | STATUS |
|---|---|---|---|---|
| **5** | **enable cross-layer prefetch** | low | **our own 0.714 recall** | **PRIMARY LEVER — NEXT** |
| 6 | wait-at-point-of-use audit | low | reasoned | NOT STARTED |
| 2 | drop `IOSQE_ASYNC` on O_DIRECT | very low | reasoned | NOT STARTED |
| 3 | sort prefetch by offset | trivial | reasoned | NOT STARTED |
| 4 | coalesce gate/up/down → 1 SQE | low-med | mechanism confirmed | **DEMOTED** — see below |
| 1 | io-wq worker cap | very low | **MEASURED** | **DEAD 2026-07-18** |
| 7 | expert co-occurrence (COUPLE) | low to measure | **MEASURED** | **DEAD 2026-07-18** |
| 8 | verify 0.97 GB/token is device-level | trivial | — | **NOT STARTED** |
| 9 | KV-stack ablation | medium | 38× compound measured on lfm2.5 | **NOT STARTED** |
| 10 | top-k prune 6→4 | medium | mass cost known (21%) | **NOT STARTED** |

---

### 1. Raise the io-wq worker cap — **DEAD, measured 2026-07-18**

> **RESULT: no effect. The kernel default bounded-worker ceiling was 16 — exactly equal to our
> `WP_IOURING_DEPTH=16`.** Sweep at 192 tok, one server per config:
> baseline 1.687 / w8 1.685 / w32 1.694 / w64 1.696 tok/s, 1.51–1.52 GB/s, 24–25% of drive —
> flat across an 8× worker change. The knob is proven to have fired:
> `wp::set_iowq_max_workers[p2p]: bounded 16 -> 64 (unbounded 63356 -> 64)`.
> Baseline also reproduced the older `bench-decode.sh` figure (1.73 / 27%), so the harness is sound.
>
> **REFRAME — the load-bearing finding: we are SUBMISSION-STARVED, not worker-capped.** The
> 16-deep ring is not even being filled. At top-6 with ~30% miss the demand path has only ~2 reads
> in flight per layer, so no ring/worker/depth tuning can fill a 6.2 GB/s drive. Only cross-layer
> speculation adds in-flight depth. Promotes item 5 to PRIMARY; demotes all queue-depth tuning.
>
> Two process notes worth keeping:
> (a) the FIRST sweep proved **nothing** — the success log was `LLAMA_LOG_INFO` and these server
> logs only surface WARN, so a silently no-op'd knob and a real no-effect are indistinguishable.
> Promoted to WARN and re-ran before concluding. **Do not declare a knob dead until its log proves
> it fired.**
> (b) **"32 thrashed" is now UNEXPLAINED** — it was not worker contention, since workers sat at 16
> and raising them is inert. Do not reuse that datapoint as evidence for anything.
>
> The knob is KEPT (default off): it is the only way to observe the kernel's io-wq ceiling, which
> is otherwise invisible.

**Original reasoning, retained for the record.** We set `IOSQE_FIXED_FILE | IOSQE_ASYNC` on *every* SQE
(`wp-file-io.cpp:243,362`, `wp-file-io-p2p.cpp:129`). `IOSQE_ASYNC` punts every read to an io-wq
**bounded worker**, so effective queue depth is capped by the worker count — **not** by
`WP_IOURING_DEPTH`. We **never call** `IORING_REGISTER_IOWQ_MAX_WORKERS` (verified: zero hits in
`src/weight-pager/`). Ring is created with `io_uring_queue_init(depth, &ring_, 0)` — flags = 0.

This explains both observations: submitted depth 16 yet 27% of drive, **and** "32 thrashed"
(worker contention, not drive saturation).

Colibrì does exactly this: `coli_uring_set_workers()` (`uring.h:84-87`) calls
`syscall(SYS_io_uring_register, fd, IORING_REGISTER_IOWQ_MAX_WORKERS, limits[2]={w,w})`, driven by
`PIPE_WORKERS` clamped to 64 (`glm.c:5853-5856`); their fastest x86 config uses `PIPE_WORKERS=16`.

- Fix: one register call after ring setup, exposed as an env knob, then sweep.
- **Evidence caveat:** the *mechanism* is confirmed in their source; the *effect* is not. No
  measured tok/s or bandwidth number exists anywhere in their repo for `URING=1`. Their README
  says PILOT "measures neutral" because their disk is already ~80% saturated — the opposite of
  our 27%. Treat as a hypothesis with a strong prior, not a validated win.

### 2. A/B dropping `IOSQE_ASYNC` on the O_DIRECT path
We added `IOSQE_ASYNC` for a real measured reason — `wp-file-io-p2p.cpp:124-126`: *"measured
`ensure_batch_submit_ms >> wait_ms`; I/O was finishing inside submit, capping random P2P at
~2 GB/s."* That fix was correct. What was never priced is the side effect: forcing io-wq
re-imposes the worker cap in item 1.

With **O_DIRECT** the original rationale largely evaporates — O_DIRECT reads are async-capable
in-kernel, so `IOSQE_ASYNC` mainly buys a thread handoff plus the worker cap. Worth testing
O_DIRECT **without** `IOSQE_ASYNC`, plus `IORING_SETUP_DEFER_TASKRUN | IORING_SETUP_SINGLE_ISSUER`
(we pass flags=0 today; Colibrì passes none either, so we would be ahead of them here).

**Do not drop it on the buffered path** without re-measuring — that is where the original
inline-completion collapse was measured.

### 3. Sort prefetch candidates by file offset before queueing
Colibrì #362: *"sort candidates by eid for sequential SSD read locality."* We dedup
(`unordered_set<int> active`, `wp-eval-cb.cpp:802`) but dedup does not imply offset ordering.
Trivial, no risk, and it is a **prerequisite for item 4** — adjacency can only be detected in a
sorted run.

### 4. Coalesce gate/up/down into one SQE per expert — **DEMOTED 2026-07-18**

> Coalescing 3 SQEs into 1 **reduces** the number of in-flight operations. Now that item 1 has
> shown we are submission-starved rather than worker-capped, fewer-but-bigger submissions is the
> wrong direction on its own. Only worth doing on top of something that actually fills the queue
> (item 5).
Confirmed gap: we issue **3 separate 4.25 MB sub-page reads** per expert (batched into one submit,
but 3 SQEs); sisters are gathered via `s_sister_cache_eb` with a cap of 18.

Two ways to do it, and the cheap one was nearly missed:
- **Runtime contiguity detection (#259)** — sort by offset, test `off + nbytes == next_off` and
  same fd, and if contiguous issue ONE aligned read (`base = off0 & ~4095`). **Requires no
  reconversion.** This is the one to do.
- **Conversion-time `merged_weight` (#362)** — new on-disk format. Rejected: a 151 GB reconversion,
  and the maintainer himself predicts realistic gains of 15-30%, not the headline 94%.

Expect a modest win (fewer SQEs, better sequentiality), **not** a 4× unlock — and note it *reduces*
in-flight op count, so pair it with item 1 rather than treating it as a substitute.

### 5. Enable the cross-layer prefetcher (already built, gated off)
`WP_PREFETCH_XLAYER` / `WP_PREFETCH_LOOKAHEAD_K` / `WP_PREFETCH_TOPK` / `WP_PREFETCH_MAX_SLOTS`
exist with a working `RouterPredictor`. **Our own measurement**, offline on `routing_capture.bin`:

| lead k | recall@6 | recall@16 | weight mass @6 |
|---|---|---|---|
| 1 | **0.714** | 0.898 | **0.795** |
| 2 | 0.656 | 0.845 | 0.744 |
| 3 | 0.592 | 0.783 | 0.686 |
| 4 | 0.569 | 0.759 | 0.662 |

Independently matches Colibrì's PILOT claim of 71.6%. **Structural argument for turning it on:**
at top-6 with ~30% miss, the demand path has only ~2 reads in flight per layer — it *cannot*
fill a 6.2 GB/s drive no matter what the ring flags are. Only cross-layer speculation adds depth.
Colibrì runs prefetch on a **second, independent ring** (`g_ub_pipe` vs `g_ub_pilot`,
`glm.c:1837`) so speculation can never delay a demand read — worth copying if ours shares a ring.

It regressed once before. That was **eviction thrash, not prediction quality** — `WP_PREFETCH_MAX_SLOTS`
is the guard to bisect first.

### 6. Audit wait-at-point-of-use vs wait-for-whole-batch
Colibrì #79: the MoE loop calls `pipe_wait(qof[j])` immediately before consuming expert *j*, not
once for the whole batch. If our pager waits for a full batch before the first matmul, this
reordering alone recovers overlap. Cheap to check.

Also audit for a **#274-genus bug**: a reserve/actual slot mismatch silently starved their cache
(`cap_for_ram` reserved 1.21 GB while only 4 slots/76 MB were used → hit rate 53%→73% once
clamped). We already found one QD-collapse bug of exactly this family (pin-before-read); a second
is plausible.

### 7. Expert co-occurrence prefetch (COUPLE) — **DEAD, measured 2026-07-18**

> **RESULT: real, but strictly dominated.** Held-out, equal budget, 720 train / 480 test:
>
> | d | B | coupled | marginal (= the pinning strategy) | lift |
> |---|---|---|---|---|
> | 1 | 6 | **0.214** | 0.135 | 1.58× |
> | 1 | 16 | 0.371 | 0.254 | 1.46× |
> | 1 | 32 | 0.510 | 0.390 | 1.31× |
> | 2 | 6 | 0.208 | 0.133 | 1.57× |
>
> The JOINT does carry signal the marginals do not — 1.58× lift, close to Colibrì's claimed 1.8× —
> so "usage is uniform, therefore prediction is hopeless" was too broad a conclusion and is worth
> remembering as such. **But** coupling at B=6 scores 0.214 against the router predictor's **0.714**
> at the same budget, and even at B=32 (5.3× the bytes) reaches only 0.510. Coupling decays less
> with depth (0.214 → 0.208) than the router does (0.714 → 0.656), yet the router still wins ~3×
> at d=2. Not worth building. Harness: `scratchpad/coupling.py`.

**Original reasoning, retained for the record.**
`COUPLE` is an offline table mapping (layer, expert) → top-16 co-activated experts of layer L+dL,
built from routing traces; at runtime a pure **table lookup** on ids the layer just produced —
no router matmul. `COUPLE_K` default 8, `COUPLE_D` 1 or 2 (`glm.c:3453-3488`).

**Why this survives our "pinning is dead" conclusion:** that conclusion is about **marginals**
(hottest expert 6.7%). Coupling is about **conditionals** — P(expert *f* at L+1 | expert *e* at L).
Load-balancing loss flattens marginals; it does **not** flatten the joint. Their measurement:
median co-activation lift **1.8×** over independence, p99 **40×**, and they claim the structure
transfers across workloads because it is a property of the model, not the session.

**Testable offline for the price of one script, using data we already have.** Their
`tools/route_coupling_report.py` methodology: equal-budget held-out simulation of marginal-frequency
prefetch vs coupled prefetch, recall of true top-8 at B=8/16/32, depth 1 and 2. Run it against
`routing_capture.bin`. If coupled recall at B=6 beats or complements our 0.714, we get more
prefetch targets at zero compute cost. If not, the idea dies cheaply.

Speculative overfetch is **nearly free for us** (bandwidth-rich, queue-poor) and expensive for
them — the regimes invert in our favour.

### 8. Verify 0.97 GB/token is device-level, not syscall-level
Before optimising further, confirm the number comes from `/sys/block/*/stat` and is not conflating
page-cache hits with real device reads. Colibrì hit exactly this and had to split
`prof_ssd_tensor_bytes()` to distinguish SSD-backed from tmpfs-satisfied reads. Our figure is from
`/proc/diskstats`, which *should* be device-level — confirm rather than assume.

Related: mirror their `iobench.c` methodology at **our** granularities (4.25 MB and 12.75 MB, thread
counts 1/4/8/16/32) to find the real knee independent of the pager. That isolates whether 1.68 GB/s
is the pager or the stack.

### 9. KV-stack ablation (not a Colibrì item — ours)
Ornith 35B-A3B measured **3.65-3.91 tok/s** at 5-way concurrency and captain 27B-dense **13.42 tok/s**
single-stream, both on the current binary. Prior ablation (2026-07-12, on lfm2.5-8b) attributed
turbo4 ≈1.8×, tiered/paged ≈2.3×, semantic index ≈5.35× — 38× compound. Those factors have never
been measured on a 35B-A3B or on DS4.

Method that worked before: **pin context depth constant** and ablate ONE flag at a time from the
real production config. Needs GPU time on mad-lab-main; mutually exclusive with murmur runs.

### 10. Top-k prune 6→4
Cuts experts fetched per layer from 6 (76.5 MB) to 4 (51 MB) = **33% less expert I/O**. Cost is
**21% of routing weight mass** (top-4 retains 0.789) — not a rounding error, needs a PPL run.

**What makes this interesting is the composition:** prefetch-from-L-1 at K=6 captures **0.795**
of routing mass — *above* the 0.789 that pruning 6→4 retains. So a prefetched set is as good as
a pruned set we would already be prepared to accept. That licenses "execute the speculated experts
instead of re-fetching them" (arXiv 2603.19289, ~14% TPOT reduction) — with the caveat below.

Predicted hits concentrate on high-weight experts (rank-1 caught 0.530 vs rank-6 0.136, a 3.9×
ratio), so misses land where pruning already says they are tolerable. **But** pruning
*deterministically* keeps ranks 1-4 while our predictor misses rank-1 **47%** of the time at K=6,
so the two are not equivalent injuries. Skip-on-miss is risky; demand-fetch-on-miss is safe.

---

## 3. Ruled out (with reasons — do not re-open without new evidence)

| item | why it is dead |
|---|---|
| **Hot-expert pinning / learning cache** | Expert usage near-uniform *by design* (load-balancing loss). Held-out coverage at our actual pool size (54/layer) is only 0.525; reaching 0.83 needs 128/layer = 63.75 GB, twice the card. |
| **CLOXCache / eviction-policy work (#223)** | Only pays when cache capacity < ~⅓ of per-token demand; we are at **8.4×**. Their own `route_sim` scores against Belady — a metric that is ~0 by construction when capacity ≥ demand. Confirmed by the PR's own data, not just our prior. |
| **O_DIRECT as a bandwidth fix** | Measured A/B: baseline 1.73 tok/s / 248.92 GB vs O_DIRECT 1.31 tok/s / 662.42 GB. O_DIRECT reads 2.66× more physical bytes because it bypasses the page cache, which was absorbing re-reads. **The page cache is a free victim tier.** |
| **MTP as a prefetch signal** | MTP's hidden is depth-43; driving shallow-layer routers with it gives **0.137** recall (chance 0.023). Only layers 41-42 clear 0.60 — 2 of 40 softmax layers. Decay is smooth and monotonic in depth distance, so it is residual-stream geometry, not a training artifact. Verified with 3 scoring arms agreeing within 0.005. |
| **DFlash→routing adapter** | 0.076 linear / 0.14 MLP against a 0.60 gate. EAGLE3 hidden is alien to the target router space. |
| **Requantisation** | DS4 is QAT'd mixed 4/8-bit. Not an option. |
| **RAID / second NVMe** | Single NVMe; the other disk is a WD Blue HDD. |
| **More host RAM** | ~$500 for DDR4 kits; the money is better spent on a second R9700 (same lever — capacity in front of NVMe — but on the fast side of the bus). |
| **`fmt=4` grouped int4 (#298)** | A **read amplifier**: scales grow ~220× in their own worked example (16 KB → 3.5 MB). Wrong direction when queue-starved, and our experts are already QAT 4.25 bpv. |
| **NUMA RAM-disk streaming (#377)** | Premise is staging weights in tmpfs on a many-socket box with spare RAM. We have 15 GB against 151 GB, single socket. |
| **Local-cluster MoE / dense sharding (#380)** | Single box. Its own diff admits sequential per-layer remote calls and no connection pooling. |
| **PR #165's disk layout** | Six objects per expert (w1/w2/w3 × weight+scale) — *worse* than our three. And #165 measures **0.59-0.85 tok/s realistic**, slower than our 1.73. It is a correctness-first CPU reference with a plain-`pread` path, not a performance target. |

---

## 4. Open questions

1. **Does the io-wq worker cap actually explain the 27% plateau?** Mechanism is confirmed; effect is
   unmeasured anywhere, including in Colibrì's own repo.
2. **Is there a second bottleneck on the PCIe hop?** Nothing in Colibrì runs a paging device on a
   GPU — their tiers are VRAM/RAM/SSD. Our "R9700 as 27.6 GB paging pool" has no analogue, so no
   external evidence speaks to queue behaviour on that hop.
3. **Does expert co-occurrence hold for DS4?** Testable offline (item 7).
4. **How much of the KV-stack cost is real on a 35B-A3B?** All 38× factors were measured on lfm2.5-8b.
5. **Does the MTP head survive 4.25 bpv as a draft?** Colibrì finds int4 MTP heads collapse to 0-4%
   acceptance, int8 gives 39-59%. Our `blk.43` is 97.5% by parameter mass at 4.25 bpv. Mitigating
   argument: DS4 is **QAT**, and draft and target share precision, so their distributions should
   agree — Colibrì's collapse was *post-training* int4 against a bf16-trained target. Instrument
   acceptance rate as the first measurement if MTP drafting is ever built.
6. **Why did cross-layer prefetch regress before?** Believed eviction thrash. Bisect
   `WP_PREFETCH_MAX_SLOTS` first.

---

## 5. Evidence-quality notes

- **No PR in Colibrì measures on AMD/ROCm.** Every number is NVIDIA (5090 / 5070 Ti) or pure CPU.
- **No measured tok/s or bandwidth exists for `URING=1`, `COUPLE`, or `PILOT_REAL`** anywhere in
  their repo. The io_uring path is unquantified.
- **#259's benchmark tables are permanently lost** (GitHub char limit, per the PR body). Its
  "3-10 GB/s" and "independent of cache size" claims rest on scrollback with no reviewer verification.
- **#362's 94.3% hit rate is an aggregate** of merged-I/O + PILOT + EMA + pinning; `merged_weight`
  is not separately benchmarked. The maintainer predicts 15-30% realistic and refuses to merge
  without flagship validation.
- **#342's +6-8% was measured on a fully resident 6×5090 rig with no NVMe in the loop**, and the
  author notes the GPU was never the bottleneck at S=1 (1.1% SM utilisation).
- **Colibrì's ring setup is *less* configured than ours** — `io_uring_setup` with a zeroed
  `io_uring_params`: no SQPOLL, IOPOLL, COOP_TASKRUN, SINGLE_ISSUER, registered buffers, or
  multishot. We are ahead on ring setup; there is no missed flag to copy.
- **Their `.coli_ssd` probe is single-threaded `pread` with random offsets** — it measures QD1
  latency, not saturation bandwidth. Do not compare it to our 6.2 GB/s saturated figure.

---

## 6. Provenance

- Our measurements: `~/wp_logs/accounting/adapter/` (`bench-decode.sh`, `nvme_replay.py`),
  and scratchpad harnesses `depth_transfer2.py`, `lookahead.py`, `prune_prefetch.py`,
  `mass_storage.py`, `hotset.py`, `extractability.py` — all offline replays of
  `~/wp_logs/accounting/routing_capture.bin`, CPU-only.
- KG: `791cb0d3` (MTP prefetch dead), `4c74845f` (scoring correction), `3f7ff065` (adjacent-layer
  reversal + I/O reframe), `d39febf0` (adapter negative), `371d9dea` (O_DIRECT dead).
- Colibrì: https://github.com/JustVugg/colibri — `c/glm.c`, `c/uring.h`, `c/tier.h`, `c/st.h`,
  `c/iobench.c`, README; PRs #79, #165, #259, #274, #279, #298, #342, #362, #377, #380, #386.
