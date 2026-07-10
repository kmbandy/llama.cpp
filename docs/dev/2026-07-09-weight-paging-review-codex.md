# Weight Paging Review: DeepSeek 4 Flash Decode

Date: 2026-07-09

Scope: the custom weight-paging implementation and its use for single-stream
DeepSeek 4 Flash decode. This is not a general llama.cpp review. The review is
based on the current source tree, the recorded benchmark logs, the current-state
document, the design notes, and the prior agent/model reports in this fork.

## Executive verdict

The project has already established the most important fact about this workload:
it is not primarily an eviction-policy problem. The measured decode trace needs
about 1.000 GB of expert payload per generated token, with roughly 224 page
misses per token. The realizable cache policies are close to one another, while
even Belady still has a large miss count. Most of the remaining traffic is
compulsory or becomes knowable too late to hide with a conventional cache.

The present 1.5-2.0 token/s result is credible for the measured workload. A
transport-only improvement to 7 GB/s will not reach 5 token/s if I/O remains
serial with target compute. With a roughly 180 ms/token non-I/O floor, one GB of
serial I/O at 7 GB/s gives approximately 323 ms/token, or 3.1 token/s. Reaching
5 token/s therefore requires at least one of these structural changes, and most
likely a combination:

1. Overlap most expert I/O with useful target compute.
2. Reduce expert bytes per token, for example top-k reduction and/or lower-bit
   expert weights.
3. Increase the fraction of experts that can be predicted early enough to be in
   VRAM before demand.
4. Change the storage representation and transport so the hardware can sustain
   close to its useful bandwidth on this access pattern.

My strongest recommendation is a two-track program:

- Build a trusted host `O_DIRECT -> pinned RAM -> asynchronous H2D` baseline and
  eliminate the currently unexplained physical-read amplification.
- In parallel, prototype a real direct-storage path using a fixed, aligned expert
  page pack and a cooperating NVMe/GPU transport such as ROCm XIO. Do not treat
  the existing dma-buf mmap path as true NVMe-to-VRAM peer DMA.

For prediction, the token-ID DFlash oracle cannot cover enough of the target
network. It can resolve the hash-routed layers, but most layers route from hidden
state. The most promising next experiment is to use DFlash hidden states as
proxies for the target router inputs, evaluate target router weights on those
states, and train small layer- or segment-specific adapters only where the direct
projection is insufficient.

The 5 token/s goal is difficult but not contradicted by the measurements. It is
not achievable by making the existing serial pager modestly faster. It becomes
plausible if the project combines near-device bandwidth, high-recall early
prediction, overlap, and some byte reduction.

## What is already strong

### The measurements have narrowed the problem

The current-state report records payload, page-in count, effective pager
bandwidth, device-level bandwidth, and token latency over useful decode windows.
That is much more valuable than optimizing from aggregate token/s alone. In
particular:

- A 128-token run records 28,727 page-ins and about 128.02 GB of pager payload.
- Cache capacity is explicitly bounded: approximately 6,500 slots fit, while
  7,000 slots fail on the tested machine.
- The LRU/Belady comparison shows a real theoretical bound rather than assuming
  another eviction heuristic will solve compulsory traffic.
- The speculative DFlash run records both its acceptance quality and its paging
  cost. A 76% acceptance result is not confused with a throughput win when it
  causes 2.18 times the page-ins.
- The btrfs compression hypothesis was tested with an uncompressed copy and was
  not supported.

These results justify moving effort away from small cache-policy variations and
toward transport, route lead time, and bytes per token.

### The pager has useful observability and fallback behavior

The pager exposes per-run counters for page-ins, batching, wait time, speculative
activity, and routing behavior. The I/O implementations also have fallbacks,
which has made it possible to compare buffered dma-buf mapping, `O_DIRECT` host
staging, and synchronous behavior without replacing the whole pager.

The design also correctly treats the gate/up/down tensors for one expert as
sisters. That semantic grouping is the right unit for prediction and for a
future storage layout, even though the current GGUF placement leaves the sisters
physically separated.

### The project is testing hypotheses instead of preserving them

Several attractive ideas have already been rejected or bounded by data:

- Increasing cache size is limited by VRAM and does not remove the compulsory
  working set.
- Alternative realizable eviction policies did not approach Belady.
- Full speculative verification improved prediction quality but made paging and
  throughput worse.
- A standalone `O_DIRECT` reader reaches about 6 GB/s, demonstrating that the
  SSD and basic block path can outperform the integrated pager.

This makes the next work more focused. The remaining gap is not simply that the
SSD is slow.

## Current architecture assessment

At a high level, the implementation does the following:

1. Expert tensors are registered as pager pages, with gate/up/down sister
   relationships.
2. A GPU pool holds a bounded number of active expert pages.
3. The graph evaluation callback discovers selected experts at routed matrix
   multiplication boundaries.
4. `ensure_batch()` allocates/evicts slots, schedules reads, waits as necessary,
   patches tensor data pointers, and updates paging state.
5. Optional predictors try to submit future pages before exact demand.

The architecture is viable for experimentation, but the control plane has become
too concentrated. `wp-pager.cpp` and `wp-eval-cb.cpp` jointly contain transport
selection, routing observation, cache policy, multiple predictors, environment
configuration, pointer patching, metrics, and fallback behavior. That makes A/B
results harder to trust because an experimental mode can change several parts of
the system at once.

The next major revision should separate four interfaces:

- **Page store:** maps `(model, layer, expert, role)` to immutable byte ranges and
  verifies their identity.
- **Transport:** submits aligned byte ranges to a destination and reports actual
  requested, completed, and failed bytes.
- **Residency manager:** owns slots, pinning, eviction, and page state.
- **Predictor:** emits page sets with a prediction window, confidence, source,
  and deadline, without directly changing cache policy.

Configuration should be captured once per context in an immutable structure,
not repeatedly read from process-global environment variables and static caches.
A small, explicitly named trusted-baseline mode should disable every predictor
and advisory reader.

## Critical findings

### 1. The dma-buf path is not demonstrated NVMe-to-VRAM P2P DMA

This is the most consequential finding because it changes the interpretation of
the 3.4 GB/s ceiling.

`src/llama.cpp:279` calls `dup_clear_o_direct()` for the file descriptors passed
to paging. The P2P I/O implementation then window-mmaps the exported VRAM
dma-buf and calls `io_uring_prep_read()` with that CPU virtual address
(`src/weight-pager/wp-file-io-p2p.cpp:93-129`). In other words, the current path
is a buffered filesystem read whose destination is a CPU mapping of the GPU BAR.
It avoids a separately allocated host bounce buffer and an explicit HIP copy,
but it does not show that the NVMe controller is issuing peer DMA writes into
VRAM.

The probe in `tools/wp-dmabuf-probe/main.cpp` establishes that HIP memory can be
exported, `fstat`ed, and `mmap`ed. It does not establish PCIe topology support,
NVMe P2PDMA addressability, block-layer acceptance of those pages, data
coherency, or end-to-end device DMA.

Linux's P2PDMA documentation describes the required provider/client/orchestrator
cooperation and the limitations on passing peer mappings through normal I/O
APIs. A raw MMIO/dma-buf user mapping is not, by itself, a general-purpose P2P
buffer for `read(2)` or `O_DIRECT`:
[Linux PCI Peer-to-Peer DMA Support](https://docs.kernel.org/driver-api/pci/p2pdma.html).

I recommend renaming this implementation and its metrics to something precise,
such as **buffered BAR-copy transport**. `direct_to_device()` and log messages
that say "P2P enabled" currently overstate what was proven.

The roughly 3.4 GB/s device ceiling is also plausible for this software path
without implicating the SSD. Each miss can involve a filesystem/page-cache read,
a CPU copy into a GPU BAR mapping, io_uring worker scheduling forced by
`IOSQE_ASYNC`, and dma-buf window-map management. CPU writes through a BAR
mapping need not approach ordinary DRAM copy bandwidth. The window cache is
small relative to the 6,500-slot pool, so a low-reuse trace can also cause
frequent `mmap`/`munmap`; existing submit-wall measurements suggest that mapping
churn is secondary, but it should still be counted directly. Profile CPU cycles,
io-wq worker utilization, faults, map operations, and PCIe write traffic before
tuning queue depth on this transport.

There is also a coherency risk. The implementation does not perform a dma-buf
CPU access synchronization protocol before the GPU consumes the mapping. The
tested platform may make the writes visible in practice, but successful output
is not a portable coherency contract. Add per-page checksums in a diagnostic mode
and an explicit synchronization design before treating this as a correctness-
preserving transport across devices and drivers.

### 2. Five token/s is a latency-overlap problem, not just a bandwidth problem

Using the measured 1.000 GB/token payload and an estimated 180 ms/token compute
floor, a simple serial model gives:

| Useful paging bandwidth | I/O time/token | Serial total | Ceiling |
| ---: | ---: | ---: | ---: |
| 2.2 GB/s | 455 ms | 635 ms | 1.58 token/s |
| 3.4 GB/s | 294 ms | 474 ms | 2.11 token/s |
| 5.0 GB/s | 200 ms | 380 ms | 2.63 token/s |
| 6.0 GB/s | 167 ms | 347 ms | 2.88 token/s |
| 7.0 GB/s | 143 ms | 323 ms | 3.10 token/s |

At 5 token/s, the total budget is 200 ms/token. Current payload fits under that
budget at 7 GB/s only if almost all of the I/O is concurrent with compute. The
fully overlapped ideal is `max(180 ms compute, 143 ms I/O)`, or approximately
5.56 token/s. This is an ideal ceiling, not a forecast; it excludes routing lead
time, synchronization, cache pollution, and tail latency.

Byte reduction makes the target less brittle:

| Scenario at 7 GB/s | Approx. serial total | Ceiling |
| --- | ---: | ---: |
| Current top-6 payload | 323 ms | 3.10 token/s |
| Top-4, same precision and compute | 275 ms | 3.63 token/s |
| Top-4 plus about half expert bytes | 228 ms | 4.39 token/s |
| Same byte reduction with 150 ms compute | 198 ms | 5.05 token/s |

The exact values must be remeasured after quantization or top-k changes, but the
direction is unambiguous. A credible 5 token/s plan needs explicit targets for
all three terms: bytes/token, useful GB/s, and overlap/compute.

### 3. The integrated `O_DIRECT` amplification is the highest-value immediate bug

The standalone direct-I/O microbenchmark sustains roughly 5.9-6.4 GB/s, while
the integrated host-bounce mode delivers roughly 1.55-1.70 GB/s of logical
pager payload. One recorded 128 GB pager run was associated with approximately
356 GB of device reads, about 2.79 times the logical payload. Copying the model
to uncompressed btrfs did not remove the effect.

Before making transport conclusions from that number, run a clean, process-
scoped experiment:

1. Disable `WP_FADVISE_LOOKAHEAD`, `WP_NEXT_LAYER_PREFETCH_K`,
   `WP_SAMPLE_ORACLE`, `WP_DRAFT_PREFETCH`, `WP_STICKY_SPEC`, and dense/advisory
   prefetch. Confirm every disabled mode in the startup log.
2. Start counters after model load and warmup; stop them immediately after the
   measured decode window.
3. Count the exact aligned bytes submitted and completed by the transport,
   including alignment prefixes and retries. Do not infer this only from logical
   page payload.
4. Record `/proc/<pid>/io` (`read_bytes` and `rchar`) for the process and obtain a
   block trace (`blktrace`/`blkparse` or an equivalent eBPF trace) with LBA, size,
   queue, and timestamp.
5. Save the exact pager offset sequence and replay it through the standalone
   reader with the same queue depth, buffer count, alignment, and request
   coalescing.
6. Compare unique requested ranges, duplicate ranges, block-layer ranges, and
   completed CQ bytes. This will distinguish application resubmission, alignment
   overhead, filesystem behavior, and unrelated device traffic.

The experiment should produce an accounting identity, not another aggregate
ratio. Until it does, the host direct path has not been fairly evaluated.

If amplification is fixed, a rotating set of large registered pinned buffers
with asynchronous H2D copies is the pragmatic near-term baseline. PCIe H2D copy
adds work, but this path is conventional, observable, and can overlap disk read,
copy, and compute. It may outperform buffered writes through a CPU mapping of
VRAM even though it contains an explicit copy.

### 4. Current DFlash/token-oracle metrics do not measure useful prediction

Several counters inflate apparent prediction quality:

- `record_active_expert_pages()` expands demand to all three sisters and is
  called at each of the three routed `MUL_MAT_ID` roles. The same logical expert
  demand can therefore be counted repeatedly.
- `draft_tid2eid_hits_in_ensure` can count a predicted sister again as the
  gate/up/down operations are ensured. Recorded hit counts near three times the
  predicted count are evidence of accounting duplication, not a hit ratio.
- `last_hit_ratio_` divides that repeated hit delta by the prediction count
  (`wp-pager.cpp:2607-2608`), so it can exceed 1.0 and drive adaptive behavior
  from an invalid statistic.
- True positives are deduplicated differently from actual-demand/false-negative
  counts, so reported precision and recall do not share the same population.
- A prediction is counted even when the page is already resident or no I/O was
  submitted. Such a prediction cannot hide an SSD miss.

Replace these with per-window, unique-page accounting. For every prediction
source, record:

- predicted pages and predicted bytes;
- pages cold at prediction time;
- I/O submitted;
- I/O completed before first demand;
- demanded before eviction;
- unused prefetched pages and bytes;
- demand-before-completion count and wait duration;
- prediction-to-demand lead-time distribution;
- wait time and bytes actually saved versus a no-predictor A/B run.

Report hash-router and learned-router predictions separately. The optimization
target should be **cold-page recall subject to a false-prefetch byte budget**, not
top-k agreement over all experts.

### 5. The DFlash hooks have inconsistent defaults and an off-by-one semantic risk

`llama_wp_on_draft_tokens()` defaults draft prefetch off unless
`WP_DRAFT_PREFETCH=1`, while `llama_wp_draft_oracle_should_run()` defaults it on
unless `WP_DRAFT_PREFETCH=0` (`src/llama-model.cpp:3354-3422`). With the variable
unset, the server may run DFlash work while the pager hook discards the result.
This should be one context-owned Boolean with one startup log.

In the strip-all-draft mode, the target consumes the token that has already been
sampled. DFlash's first newly generated token predicts the following output, not
the target's current input token. For token-ID-to-expert hash routing, the sampled
token is the exact current oracle; the first DFlash output is a candidate for the
next target step. Treating it as the current token shifts the prediction window.

Make the temporal contract explicit in code and telemetry:

- target input token `t`;
- DFlash state conditioned through `t`;
- pages predicted for target evaluation of `t`;
- DFlash output token proposed for target evaluation of `t+1`;
- deadline and first target node at which each page is required.

### 6. The sample-token oracle is normally inert in steady state

With `WP_SAMPLE_ORACLE_EVICT=0`, the oracle only consumes free slots. Once the
6,500-slot pool is full, recorded baseline runs show zero oracle page submissions.
It can mark or score pages but cannot initiate the I/O needed to turn correct
knowledge into overlap. Enabling oracle eviction without accurate accounting is
also risky because false or late predictions can evict current working pages.

The residency manager needs explicit speculative reservations and cancellation:

- cap speculative slots and bytes per prediction window;
- never evict pages pinned by the current layer;
- prefer evicting unused speculative pages before demand-loaded pages;
- cancel obsolete queued work where the transport supports it;
- make completed-but-unused speculative bytes visible as a first-class cost.

### 7. Token IDs cannot predict most of this network's routing

The token-ID mapping is exact only for the hash-routed layers. The deeper routed
layers select experts from hidden-state-dependent router inputs. Even perfect
token-ID prediction therefore covers too little of the roughly one GB/token
traffic to meet the overlap requirement.

The next DFlash experiment should use hidden states:

1. Capture target pre-MoE router inputs and actual selected experts for a
   representative corpus, including whether each page was cold at prediction
   time.
2. Capture DFlash hidden states at its five layers for the same token positions.
3. Align DFlash stages with target layer segments, initially around the existing
   target taps `[3, 13, 23, 32, 42]`.
4. Apply the target layer's router weights directly to the aligned DFlash hidden
   state and measure top-k and cold-page recall at different expansion budgets.
5. Only if direct projection is insufficient, train small linear/low-rank
   per-segment adapters from DFlash hidden state to target router input or logits.
6. Evaluate one- and multi-layer lead times, useful completed bytes, false bytes,
   and end-to-end target token/s. Do not begin with full speculative target
   verification.

The predictor likely needs high recall rather than perfect precision. A simple
overlap model illustrates the bar: at 7 GB/s, about 0.86 recall of current cold
bytes leaves only 20 ms of demand I/O. If predictions cover that much traffic,
their precision must still be high enough that predicted bytes fit within the
compute window; around 0.68 precision at 0.86 recall is a rough lower bound for
one GB/token and 180 ms of compute. Tail latency and queue contention will make
the real requirement stricter.

### 8. Missing active pages can silently select the wrong expert

In the evaluation callback, null expert pointers are replaced with
`first_active_slot` when one is available (`src/weight-pager/wp-eval-cb.cpp:721-
884`). This avoids dereferencing null, but it can cause an active missing expert
to execute another expert's weights. That is silent model corruption.

An allocation or page-in failure for an active expert must fail the evaluation
with a precise error. If inactive expert entries require a valid pointer because
of kernel implementation details, use a documented sentinel only after proving
that the kernel never reads inactive entries. Keep active and inactive pointer
states distinct in the callback.

This is a release-blocking correctness issue independent of performance.

## Transport path forward

### Near term: make host direct I/O the reference implementation

The reference path should be intentionally unsurprising:

- immutable aligned page store;
- `O_DIRECT` reads into registered pinned buffers;
- multiple queues or workers as required to reach device bandwidth;
- explicit asynchronous H2D copies on dedicated streams;
- event-based slot readiness and compute dependencies;
- exact process-level and transport-level byte accounting.

Use two or three pipeline stages so NVMe read, PCIe copy, and GPU compute can run
concurrently. Measure queue occupancy and p50/p95/p99 page-batch latency, not only
aggregate GB/s. This becomes the correctness and throughput baseline against
which any direct-storage path must win.

### Medium term: use a paging-oriented expert store

GGUF is a good model distribution format, but the original tensor layout is not
ideal for a storage pager. Gate/up/down ranges for the same expert are separated,
which turns one logical expert into three reads and impairs coalescing.

Create a generated expert page pack with:

- gate, up, and down bytes interleaved contiguously per `(layer, expert)`;
- fixed 4 KiB-or-better alignment and stable offsets;
- a manifest mapping original tensor identity, quantization, shape, and checksum
  to each range;
- a source-model hash and pack-format version;
- checksums that can be enabled during validation;
- no compression, reflinks, relocation, or mutable allocation after creation;
- request sizes chosen from measured NVMe behavior rather than page abstraction
  alone.

One expert sister group is about 13.3 MiB in the current model. The transport may
split that at the controller's optimal request size or MDTS, but the store should
make the ranges adjacent so those decisions are possible. This layout reduces
18 independent role reads per six-expert layer to six logical expert ranges. It
does not reduce bytes, but it removes avoidable filesystem and submission work.

The pack can remain a normal file for the host `O_DIRECT` path. A raw-device
transport can use the same manifest with explicit LBAs or a dedicated immutable
partition. Do not depend on translating the live extents of a compressed or
copy-on-write filesystem during decode.

### Experimental direct storage: ROCm XIO

ROCm XIO is materially closer to the desired architecture than mmaping an
exported dma-buf. Its NVMe endpoint supports GPU-initiated command submission and
VRAM destinations while bypassing the normal kernel block layer:
[What is ROCm XIO?](https://rocm.docs.amd.com/projects/rocm-xio/en/beta-0.1.0/what-is-xio.html)
and [XIO endpoints](https://rocm.docs.amd.com/projects/rocm-xio/en/beta-0.1.0/reference/endpoints.html).
Its memory-mode documentation describes dma-buf-backed device memory and physical
address resolution through the XIO kernel component:
[XIO memory modes](https://rocm.docs.amd.com/projects/rocm-xio/en/beta-0.1.0/conceptual/memory-modes.html).

This is the right kind of cooperating stack to prototype, but it is beta software,
not a drop-in production answer. The published documentation notes RDNA doorbell
coherency considerations and verification caveats for some device-memory I/O
cases. Its published performance examples also do not prove sustained throughput
for this model's 4.45 MiB pages:
[XIO performance](https://rocm.docs.amd.com/projects/rocm-xio/en/beta-0.1.0/reference/performance.html).

Prototype gates should be:

1. End-to-end checksummed reads into VRAM over millions of randomized aligned
   ranges with no corruption.
2. A replay of the real recorded expert offset sequence at several queue depths.
3. Sustained useful bandwidth and p99 latency compared with the pinned-host
   reference.
4. Correct behavior under cancellation, short reads, reset, and process exit.
5. A topology check proving the NVMe and GPU can communicate through a supported
   PCIe path.

Validate topology with `lspci -tv`, relevant `lspci -vv` capabilities, IOMMU/ACS
configuration, Resizable BAR, and the transport's own P2P-distance checks. A
successful dma-buf mmap probe is not a topology test.

Do not add a second SSD until the software consumes most of one drive's measured
bandwidth on the production access sequence. Striping becomes useful after the
single-device queue and prediction pipeline are no longer the bottleneck.

## Exact overlap opportunities without a learned predictor

The current exact expert route is handled at the routed `MUL_MAT_ID` callback,
after selection is available. The scheduler also has a post-computation callback
phase (`ask=false`) around requested routing boundaries, but the weight-paging
callback currently returns without using that phase.

A bounded experiment is to split route discovery from expert execution:

1. Stop after the selected-expert indices are produced.
2. At the post-routing callback, immediately submit all gate/up/down sisters for
   the exact route.
3. Execute any routing-independent shared-expert or other independent work.
4. Wait only when the routed expert operation reaches its true data dependency.

This does not create large lead time, and shared-expert work may be too small to
move throughput materially. It is nevertheless a clean correctness-preserving
measurement and establishes the best overlap available without prediction.

The current same-expert next-layer lookahead (`WP_NEXT_LAYER_PREFETCH_K`, default
1) should be off in the trusted baseline. Future layer routing depends on the
current layer output, and prior traces show little useful benefit from copying
the same expert IDs forward. Already-resident predictions can look successful
without hiding any I/O.

## Recommended experiment sequence

### P0: correctness and measurement

1. Replace active missing-page pointer substitution with a hard evaluation
   failure.
2. Unify DFlash enablement and token-position semantics.
3. Repair predictor metrics around unique cold pages and completed-before-demand
   bytes.
4. Add a single trusted-baseline configuration and print its complete resolved
   configuration at startup.
5. Make `test-weight-pager` complete reliably and ensure its async I/O tests do
   not silently fall back when the intended backend is unavailable.

### P1: explain and recover physical bandwidth

1. Run the clean `O_DIRECT` accounting experiment.
2. Replay the exact request trace outside inference.
3. Add a packed expert store and rerun both replay and inference.
4. Pipeline registered pinned-buffer reads and H2D copies.
5. Record a latency decomposition per token: target compute, prediction compute,
   storage wait, H2D wait, callback/patching overhead, and overlap.

Success gate: at least 5.5 GB/s useful transport bandwidth on the real offset
trace, with physical bytes within 10% of submitted aligned bytes and no checksum
failures.

### P2: turn DFlash into an I/O predictor

1. Produce the aligned DFlash/target hidden-state routing dataset.
2. Establish direct target-router-on-DFlash-state baselines.
3. Train only small adapters where necessary.
4. Sweep prediction expansion, lead time, and speculative byte caps.
5. Integrate the predictor without speculative target verification.

Success gate: at least 85% recall of cold demanded bytes, enough precision to
fit prediction traffic inside the available transport window, and a measured
reduction in demand wait on an untouched evaluation corpus.

### P3: reduce bytes and combine

1. Evaluate runtime top-6 to top-4 routing with perplexity/task-quality gates.
2. Evaluate a lower-bit expert-only quantization while preserving more sensitive
   dense/router/shared tensors.
3. Combine the best quality-acceptable byte reduction with prediction and
   pipelined transport.
4. Tune cache policy only after speculative traffic changes the observed access
   distribution.

Success gate: 200 ms/token or less over a long, steady-state, single-stream run,
with quality deltas reported against the current top-6 model.

### P4: direct-storage prototype

1. Validate ROCm XIO or another cooperating direct-storage stack on the exact
   machine topology.
2. Read the immutable expert pack by raw block range into VRAM.
3. Replay the production trace with checksums and failure injection.
4. Compare end-to-end inference with the pinned-host pipeline.

Keep this path only if it produces a meaningful end-to-end latency gain after
prediction and byte reduction. Peak microbenchmark GB/s alone is not sufficient.

## Proposed next major version

A next major weight-paging release should be framed as a deadline-aware expert
streaming subsystem, not just a larger cache. I would slate these features:

### Stable page-store format

- Versioned, checksummed, immutable expert page pack.
- Co-located sister tensors and storage-optimal alignment.
- Offline pack builder and verifier.
- Explicit compatibility with host direct I/O and raw/direct-storage transports.

### Pluggable, measurable transports

- Pinned-host `O_DIRECT` reference transport.
- Optional XIO/direct-storage transport behind a strict capability probe.
- Unified async submission, completion, cancellation, and error contracts.
- Exact byte, latency, queue-depth, and checksum telemetry.

### Deadline-aware prediction

- Prediction windows tagged by target token, layer range, and deadline.
- Separate hash-token and hidden-state predictor sources.
- Cold-page recall/precision and useful-prefetch reporting.
- Speculative byte/slot budgets with cancellation and pollution controls.
- Offline trace replay and predictor evaluation tools.

### Pipeline-aware scheduling

- Exact post-route prefetch boundary.
- Explicit disk, copy, and compute events rather than callback-wide waits.
- Per-slot readiness dependencies.
- Shared-expert and other independent work scheduled into unavoidable I/O gaps.

### Correctness and operability

- No silent expert substitution on pager failure.
- Page checksums and optional sampling in production.
- One typed configuration object with a printed resolved configuration.
- Deterministic trusted-baseline mode.
- Long-run fault, reset, out-of-space, and cancellation tests.

## Items I would stop or defer

- Do not spend another major cycle on small LRU variants unless prediction
  materially changes the trace. Existing evidence says the current demand trace
  is not policy-limited.
- Do not call the dma-buf mmap reader true P2P or infer NVMe DMA behavior from
  its throughput.
- Do not use total predicted-expert overlap as the predictor objective. Resident
  predictions and repeated sister counts do not save I/O.
- Do not use full target speculative verification as the main paging solution.
  Its acceptance rate is promising for generation, but the measured paging cost
  is counterproductive for this objective.
- Do not add storage hardware before resolving the integrated-versus-standalone
  bandwidth gap.
- Do not combine top-k, quantization, prediction, transport, and cache changes in
  one run. Preserve isolated A/Bs and a trusted baseline.

## Validation performed for this review

I built and ran `test-weight-pager` with:

```text
CCACHE_DISABLE=1 cmake --build build-hip --target test-weight-pager -j4
./build-hip/bin/test-weight-pager
```

The target built successfully and many pager tests passed, but the process
aborted in `test_routing_boundary_prepass` with `free(): invalid pointer` and
exit status 134. The io_uring unit paths also reported that queue initialization
was not permitted in this environment and exercised fallbacks instead. The test
result is therefore not clean, and the intended asynchronous paths were not
fully validated here.

The worktree already contained extensive uncommitted weight-paging changes. This
review did not alter those source files.

## Bottom line

The weight pager is beyond the stage where another cache heuristic or a higher
microbenchmark number is likely to deliver the goal. The project now needs a
measured streaming architecture:

- make every physical byte explainable;
- use an immutable, contiguous expert store;
- establish a fast pinned-host reference transport;
- prototype a real cooperating direct-storage path;
- predict hidden-state-dependent routes early enough to overlap them;
- reduce bytes where quality permits;
- make failures loud and predictor benefit measurable.

The fastest credible route to 5 token/s is likely top-k or expert-bitwidth
reduction plus high-recall hidden-state route prediction over a 6-7 GB/s
pipelined transport. Any one of those improvements in isolation is unlikely to
be enough.
