# Design: worker-side phase instrumentation + per-worker wait attribution

Status: design, ready for implementation
Author: Claude (design/review). Implementation to gpt-5.6-terra.
Date: 2026-07-30

## 1. What we know, and what we do not

Measured tonight with `WP_DISPATCH_STATS=1` (GLM-5.2, 75 routed layers, 32 tokens):

```
layers=75 pack=0.36 ms issue=4.41 ms wait=925.87 ms unpack=0.18 ms total=933.65 ms
first_await_in_flight avg=2.6 (workers=3)
spine eval = 926.15 ms/token, end-to-end 0.914 tok/s
```

Dispatch `total` (933.65 ms) ≈ the entire eval time (926.15 ms), and `wait` is **99.2%**
of it — **12.34 ms per layer spent blocking on worker responses.**

**Two hypotheses are already dead. Do not revive them:**

- *Scheduler barriers.* I modelled ~9.5 ms/layer of fixed cost outside the dispatch op
  (`ggml_backend_sched` draining the GPU graph twice per layer over Thunderbolt 3).
  **That gap measures ~0.** The op accounts for essentially all of decode.
- *Cross-worker serialization.* `first_await_in_flight = 2.6 of 3` proves requests are
  issued before awaiting. The workers overlap and we still wait 12.34 ms/layer.

**What is unknown:** what happens inside those 12.34 ms. Prior estimates put NVMe
page-in at only ~3.5–4.2 ms/layer, leaving roughly **8 ms/layer unaccounted on the
worker side**. Do not guess the split again — three static models were wrong today.

## 2. Part A — per-worker wait attribution (spine side, do this first)

Because requests overlap, `wait` is bounded below by the **slowest** worker, not the sum.
So the first question is cheap: *is one worker setting the pace?*

The spine already knows when each response arrives. Extend `worker_dispatch_stats`
(`src/pipeline/pipe-expert-dispatcher.h:31`, currently just `endpoint` + `n_experts`):

```cpp
uint64_t ns_wait  = 0;   // issue -> this worker's response consumed
uint64_t n_requests = 0;
uint64_t n_experts_total = 0;
```

Aggregate per decode alongside the existing summary and emit one line per worker:

```
expert dispatch worker 100.86.191.92:8801   requests=N experts=M wait=XX.XX ms (avg X.XX ms/req)
expert dispatch worker 100.102.191.30:8803  ...
expert dispatch worker 100.102.191.30:8804  ...
```

**Why this is decisive.** Three outcomes, three different next steps:

| observation | meaning | action |
|---|---|---|
| one worker's avg ≫ others | that machine/backend/drive is the pace-setter | fix or rebalance that one |
| all three roughly equal | the cost is structural, common to every worker | Part B decomposition |
| R9700 fast, 2026 pair slow and equal | the two 2026 workers are contending on one SN550 | rebalance shard split |

The third case is a live possibility we have never tested: the 1070 and RX 480 share a
single SN550 at 3.08 GB/s while the R9700 has an SN850X at 6.25 GB/s to itself.

## 3. Part B — worker-side phase timer

In `tools/wp-expert-worker/wp-expert-worker.cpp`, time these phases per request with
`std::chrono::steady_clock`:

| phase | from → to |
|---|---|
| `ns_recv` | first byte of request → request fully parsed |
| `ns_lookup` | cache/slot resolution; also record `n_hit`, `n_miss` |
| `ns_read` | NVMe page-in for misses (concurrent, up to the staging buffers) |
| `ns_h2d` | staging → VRAM upload |
| `ns_compute` | GPU expert FFN + weighted sum |
| `ns_send` | response serialize + write |

Also accumulate `bytes_read`, `n_requests`, `n_experts`.

### 3.1 Do NOT add synchronization to make timing possible

This is the constraint that matters most. **If timing a phase would require inserting a
device synchronize that is not already there, do not insert it — report that phase as
unavailable instead.** A sync added for measurement serializes the pipeline and changes
the thing being measured.

We have been burned by exactly this: a previous probe (`PHASEPROBE=1`) slowed a kernel
~44× while its own banner claimed "zero memory perturbation", and produced a fabricated
counter that motivated a week of misdirected work. Use the syncs the worker already
performs before responding; anything finer is not worth a false number.

### 3.2 Reporting

Emit a **cumulative** summary line **periodically (every ~5 s)**, not per request.

Rationale: there are ~75 layers × 32 tokens ≈ 2400 requests per worker per run, so
per-request logging would both flood and perturb. And it must be periodic rather than
at-exit because **the harness may SIGKILL the worker** (both `llama-server` and the
worker were measured today to survive SIGINT, so teardown escalates to SIGKILL, which
cannot be caught). A periodic line survives being killed; an exit handler does not.

Gate behind `WP_WORKER_STATS=1`.

**Emit at LLAMA_LOG_WARN, or plain stdout.** `LLAMA_LOG_INFO` is filtered at the default
server logger threshold of 3 (libllama WARN maps to verbosity 2 and passes; INFO maps to
4 and is dropped) — this already cost us one full measurement run tonight.

## 4. Required control — the probe must not change the result

Run the identical workload twice, `WP_WORKER_STATS=1` and `=0`, and compare end-to-end
tok/s. **If throughput moves by more than ~2%, the instrumentation is perturbing the
system and its numbers must not be trusted.** Report both numbers. This is not optional
ceremony; it is the specific failure mode that has produced fake findings here before.

## 5. What we expect, stated in advance so we cannot rationalize afterwards

Writing the prediction down first, so a mismatch is a finding rather than something to
explain away:

- NVMe read ≈ 3.5–4.2 ms/layer (from measured bytes and rated drive speeds)
- if `ns_read` lands near that and `ns_compute` is small, the missing ~8 ms is protocol,
  scheduling, or H2D — and the fix is worker-side software
- if `ns_compute` is large, the experts themselves are the cost and the fix is batching
  or better kernels
- if `ns_read` is far *above* the estimate, the cache hit rate or read amplification is
  worse than believed — check for a repeat of the btrfs `compress=zstd` / O_DIRECT
  interaction that caused 2.49× amplification once already

## 6. Sequencing — Part A ships alone, Part B is gated

**Implement Part A only for now.** Part A is entirely spine-side (`src/pipeline/`,
built into `libllama` in mad-lab-main's `build-hip`) and touches nothing on
mad-lab-2026. It is zero-risk and may localize the problem on its own.

**Part B is deferred because it carries a real hazard.** It changes
`tools/wp-expert-worker`, which must be built in *both* `build-hip` (main) and
`build-army` (2026) — the machines have separate checkouts and separate build trees, so
a worker change built in only one place means the 2026 workers silently run old code.

But **`build-army` is the tree the live fleet services run from**: pid 855466 (nemotron
embedder) and pid 3025042 (llama-router) both execute
`/home/kmbandy/GitHub/llama.cpp/build-army/bin/llama-server` and load its `libllama.so`.
Rebuilding that library underneath them is exactly what crash-looped the embedder
earlier today (SEGV in 1.155 s after `libllama` was rebuilt without `llama-server`).

Before Part B is built, decide the mitigation deliberately:
- move the existing `libllama.so*` aside first so running processes keep their inode,
  rather than having the linker overwrite in place, **and**
- rebuild `libllama` and `llama-server` together, never one alone, **and**
- confirm with kmbandy before touching that tree at all.

Do not do any of this as a side effect of Part A.

## 7. Constraints
- Do not use output sha256 to verify anything took effect: greedy argmax masks small
  numeric differences, and output was bit-identical across moving 6 GiB of weights
  between devices today.
- mad-lab-2026 runs LIVE FLEET SERVICES (pid 855466 nemotron embedder, pid 3025042
  llama-router) against `build-army`. Rebuilding that tree is expected, but do not
  restart or disturb those processes.
