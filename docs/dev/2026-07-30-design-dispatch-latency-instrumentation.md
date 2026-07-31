# Design: expert-dispatch latency instrumentation + pack/unpack fix

Status: design, ready for implementation. **Depends on the failure-isolation change landing first** (same two files).
Author: Claude (design/review). Implementation to gpt-5.6-terra.
Date: 2026-07-30

## 1. The question this must answer

Four GPUs across two machines produce **0.889 tok/s** against a **0.72 tok/s**
single-process baseline — only **1.23x**. We need to know why before optimizing anything.

Measured per-layer budget (997 ms/token ÷ 75 routed layers = **13.3 ms/layer**):

| leg | overlapped | serialized |
|---|---|---|
| R9700 NVMe, ~22 MB @ 6.25 GB/s | 3.5 ms | 3.5 ms |
| 2026 pair NVMe, ~13 MB @ 3.08 GB/s | — | 4.2 ms |
| wire: 12.3 KB up + 24.6 KB down per worker, 0.6 ms RTT | ~1.5 ms | ~2.5 ms |
| **accounted** | **~5 ms** | **~10 ms** |

Nothing is saturated: R9700 averaged 1.87 GB/s against a 6.25 GB/s drive, the 2026 pair
1.09 GB/s against 3.08 GB/s, worker CPU duty ~33%. So even the pessimistic serial case
leaves **~3 ms/layer unexplained**, and the optimistic case leaves ~8 ms.

**Working hypothesis: the system is latency-bound on fixed per-layer overhead, not
bandwidth-bound.** Adding GPUs adds bandwidth to a problem that is not short of it.

> **This hypothesis is NOT established.** It rests on a model with an assumed prefill
> page count. This session has already had three confident static analyses demolished by
> measurement (the unanchored `-ot` regex, the stale-spine port hijack, and a false
> "the RX 480 is 10x slower" claim). **Measure first. Do not optimize on this model.**

## 2. Change A — bulk pack/unpack (do this regardless, it is strictly better)

`graph_dispatcher::compute` currently moves data one element at a time:

```cpp
// pack, line ~127: 6144 accessor calls per layer
for (size_t i = 0; i < wire_activations.size(); ++i)
    wire_activations[i] = (uint16_t) ggml_fp32_to_fp16(ggml_get_f32_1d(activations, (int) i));

// unpack, line ~159: 6144 more
for (size_t i = 0; i < result.size(); ++i)
    ggml_set_f32_1d(dst, (int) i, result[i]);
```

`ggml_get_f32_1d`/`ggml_set_f32_1d` switch on tensor type **per element**. The op already
guarantees `activations` is F32 (line 80) and `dst` is the same shape (line 121), so both
loops are avoidable:

```cpp
if (ggml_is_contiguous(activations)) {
    ggml_fp32_to_fp16_row((const float *) activations->data,
                          (ggml_fp16_t *) wire_activations.data(),
                          (int64_t) wire_activations.size());
} else { /* keep the existing scalar loop as the fallback */ }

if (ggml_is_contiguous(dst) && dst->type == GGML_TYPE_F32) {
    memcpy(dst->data, result.data(), result.size() * sizeof(float));
} else { /* existing scalar loop */ }
```

`ggml_fp32_to_fp16_row` is declared at `ggml/include/ggml.h:374` and is SIMD-optimised.

**Keep the scalar paths as guarded fallbacks — do not delete them.** Correctness must not
depend on a contiguity assumption that some future graph change quietly breaks.

## 3. Change B — per-leg timing instrumentation

Extend `dispatch_stats` (`src/pipeline/pipe-expert-dispatcher.h:36`), which today has
counts but **no timing at all**, and is only ever printed by a standalone test tool
(`tools/wp-expert-dispatcher/main.cpp:175`) — never in the server path.

Add, using `std::chrono::steady_clock` (monotonic; never `system_clock`):

```cpp
uint64_t ns_pack   = 0;  // F32 -> F16 conversion
uint64_t ns_issue  = 0;  // building + sending all requests
uint64_t ns_wait   = 0;  // first await -> last response  <-- THE ONE THAT MATTERS
uint64_t ns_unpack = 0;  // writing results into dst
uint64_t ns_total  = 0;  // whole compute() body
```

`first_await_in_flight` already exists (line 39) and is the overlap discriminator:

- `first_await_in_flight == n workers` → requests were all issued before awaiting: **overlapped**
- `first_await_in_flight == 1` → **serialized**, and the per-worker legs are adding rather
  than overlapping. This alone would explain several ms/layer.

### Aggregation and reporting

Accumulate per layer into the `graph_dispatcher`, then emit **one summary per decode**, not
per layer — 75 layers x 32 tokens of logging would itself perturb the measurement.

Report at minimum:

```
expert dispatch: layers=75 pack=X.XX ms issue=X.XX ms wait=XX.XX ms unpack=X.XX ms
                 total=XX.XX ms  first_await_in_flight avg=N.N  (workers=3)
```

Gate it behind an env var (e.g. `WP_DISPATCH_STATS=1`) or an existing verbosity level so it
costs nothing when off. Timer overhead must stay off the hot path when disabled.

### The number we are actually hunting

`ns_total` summed over the decode vs. the spine's reported eval time. **The gap is the
per-layer scheduler barrier cost** — time the graph spends outside the dispatch op
entirely. If that gap is ~9 ms/layer, the sched-sync hypothesis is confirmed and the fix
is architectural (§5). If instead `ns_wait` dominates, the problem is worker-side or wire
latency and the fix is prefetch/overlap. **These imply completely different next steps,
which is exactly why this must be measured before anything else is built.**

## 4. Change C — three-op threading split (DESIGNED, BUT GATED — do not build yet)

The op is currently created with `n_tasks = 1` (`pipe-expert-dispatch-graph.cpp:89`) and
asserts `nth == 1` (line 103). Threading is available: `GGML_N_TASKS_MAX` makes the CPU
backend use `n_threads` (`ggml/src/ggml-cpu/ggml-cpu.c:2485`). mad-lab-main is a 3900X
with 24 threads, currently idle during dispatch.

**Structural catch:** `ggml_barrier` is internal to `ggml-cpu.c` (line 627) and is *not*
reachable from a custom-op callback — the callback receives only `ith`/`nth`. So one op
**cannot** express *parallel pack → serial network → parallel unpack*; there is no way to
synchronize the middle. Threading requires splitting into three graph ops:

| op | n_tasks | work |
|---|---|---|
| pack | `GGML_N_TASKS_MAX` | F32 → F16 into a wire tensor |
| dispatch | `1` | network I/O only |
| unpack | `GGML_N_TASKS_MAX` | F32 result → `dst` |

**Why this is gated:** after Change A, pack/unpack is a SIMD row conversion plus a memcpy —
plausibly ~0.02 ms/layer. Threading a 0.02 ms leg across 24 threads buys nothing and adds
two graph ops plus an intermediate tensor per layer. Threads also cannot speed up
`ns_wait` (24 threads waiting on one socket is still one socket) or the sched barrier.

**Build this only if the §3 measurement shows pack+unpack is material after Change A.**
The design is recorded here so it is ready if the numbers call for it.

## 5. If the sched-barrier hypothesis is confirmed (future work, not this change)

The dispatch is a CPU op in the middle of every routed layer, so `ggml_backend_sched` must
drain the GPU graph, sync device→host, run one CPU op, and sync back — **twice per layer,
150 times per token**, on the 6900 XT which sits on a Thunderbolt 3 link. Candidate
directions, in rough order of appeal:

1. **Cross-layer pipelining** — issue layer N+1's activations before awaiting layer N.
   Bounded by the router: expert selection for N+1 depends on N's output. May require
   speculation, which the MTP head could plausibly supply.
2. **Move the op onto the spine's backend** so no host round-trip is needed per layer.
3. **Batch multiple layers per round trip** where routing allows.

All three are real design work and must not be started before §3 reports.

## 6. Constraints

- Depends on the failure-isolation change (`2026-07-30-design-dispatcher-failure-isolation.md`)
  landing first — same two files, concurrent edits would conflict.
- Rebuild `libllama` **and** `llama-server` together; an ABI change rebuilt alone
  crash-looped the fleet embedder earlier today.
- Do not use output sha256 to verify a change took effect. Greedy argmax masks small
  numeric differences — output was bit-identical across moving 6 GiB of weights between
  devices today. Verify directly.
