# Design: expert deferral as an NVMe occupancy fix

Status: design, ready for review. NOT ready for implementation — see section 2, the
quality question is unresolved and is kmbandy's call.
Author: Claude (design/review)
Date: 2026-07-31

## 1. The finding this is built on

Measured 2026-07-31, 256-token GLM-5.2 run, 4 GPUs, `/proc/diskstats` sampled every 5 s
(`qd_sample.py`, kept in the job tmp dir and at `/tmp/qd_sample.py` on main):

```
box            drive     QD      util%   MB/s avg   MB/s while busy   capability
mad-lab-2026   SN750     11.5     40      1230       ~3.1 GB/s        2.86-2.91 @ QD16
mad-lab-main   SN850X    14.0     30      1980       ~6.7 GB/s        2.95 @ QD16
```

**The queue is 11-14 deep, not 1, and while the device is busy it runs at or above its
measured random-read rating. It is only busy 30-40% of the time.** Bandwidth is lost to
gaps, not to shallow queues.

The gap has a structural cause: layer N blocks until every one of its experts has landed,
*then* layer N+1 computes its routing and issues reads. The drive drains, waits on a GPU
round trip and the next router, and bursts again. 75 times per token.

Read is the largest single term in the worker budget, so closing those gaps is the biggest
remaining lever.

### 1.1 What this design is NOT, and two corrections

**Not latency-hiding behind compute.** KTransformers (SOSP'25) overlaps expert fetch with
the *next layer's attention*. We have no attention to hide behind: dispatch total
(933.65 ms) is essentially the entire eval time (926.15 ms), and GPU submit is 0.946 s of a
36 s run — 2.6%. Their accounting assumes a single node with real compute in the shadow.
Ours has neither. Do not port their recipe; re-derive it.

**Not an I/O-concurrency fix.** That was my framing earlier the same day and the measurement
refutes it — concurrency *within* a burst is already fine at QD 11-14. This is an
**occupancy** fix. The overlap we want is layer N's deferred fetches against layer N+1's
**fetches**, not against its compute.

## 2. THE UNRESOLVED QUESTION — read this before building anything

**Expert deferral is an approximation. It changes model output.**

The technique defers the lowest-router-weight experts of layer N and folds their contribution
into layer N+1's residual stream. It works because the residual is dominated by the identity
path, so a one-layer-late addition is a small perturbation — but it is a perturbation, and it
is not free.

This is a different class of change from everything else we have done on the pager. Every
prior lever was numerics-preserving: the O_DIRECT alignment fix, the btrfs `chattr +C`, the
bulk pack/unpack, the batched graph, the prefill broadcast fix. Each was "same math, less
waste". **This one trades output quality for throughput**, and that trade is kmbandy's to
make, not mine.

Before implementation, we need a decision on:
- Is a measurable quality cost acceptable at all for this workload?
- If yes, what is the budget? (e.g. "PPL within +0.05 of baseline" is a gate; "output still
  looks coherent" is not — greedy argmax masks real numeric drift, which is why output
  sha256 was bit-identical across moving 6 GiB of weights between devices.)

If the answer is "no quality cost", stop here and go to prefetch-behind-DSpark instead, which
is exactly lossless. Section 6 covers the non-lossy alternative that is available today.

### 2.1 RESOLVED — the quality budget (kmbandy, 2026-07-31)

Some quality loss is acceptable, "as long as we're not talking about absurd numbers".

**THE GATE: <= 1% PPL increase on wikitext, paged-with-deferral vs paged-without, same
config, only K varying.**

- Measure as a **sweep over K**, not a single point. Defer-1-of-8 should be nearly free;
  KTransformers' published setting is closer to defer-6-of-8. The curve locates the knee, and
  a sweep costs little more than a point since loading the model is the expensive part.
- The comparison MUST be paged-vs-paged. Do not compare against a resident baseline: the
  0.95% HIP paged-vs-resident PPL gap is still un-characterised, and folding that confound in
  makes both measurements uninterpretable.
- PPL, not coherence. Greedy argmax masks real drift — output sha256 was bit-identical across
  moving 6 GiB of weights between devices.

### 2.2 DO NOT develop this on DeepSeek-V4-Flash

kmbandy's expectation is that DS4-Flash at native FP4/FP8 would show less deferral damage than
GLM-5.2 at UD-Q2_K_XL. That may be right, but three things argue the other way and none are
settled:

1. **Deferring k of 6 is a bigger bite than k of 8.** DS4 routes `num_experts_per_tok: 6`;
   GLM-5.2 routes 8. Same K discards a larger share of routed capacity.
2. **Hyper-connections make "the next layer's residual" ambiguous.** DS4 has `hc_mult: 4` —
   `H_pre` routes each layer to ONE of four residual streams. Deferral assumes a single
   identity path that a late contribution can safely join. With four streams and a learned
   selector, a deferred partial may land in a different stream than the one it was computed
   for. This question does not exist for a plain residual model and MUST be answered before
   deferral touches DS4.
3. **This model family is measurably fragile.** The Sinkhorn fusion A/B found DS4-Flash
   flipping between degenerate repetition and coherent prose on a 1.2e-07 numerical
   difference (ce4764e71, 2026-07-19, still unexplained). Chaotic amplification produces
   different-but-coherent text; that produced a sick output.

Argument in the other direction: DS4 has `n_shared_experts: 1`, an always-active expert
carrying a baseline contribution independent of routing. That is a genuine stabiliser. Whether
it outweighs the three above is empirical.

**Therefore: build and gate deferral on GLM-5.2**, where a baseline and a working rig exist.
Carrying an unvalidated approximation into the DS4 pivot at the same time as a new
architecture, a new quantisation format, a 314-commit upstream sync and DSpark means that if
quality moves, nothing identifies which of the five caused it.

## 3. What to build (if section 2 clears)

Per layer, in the dispatcher:

```
route layer N -> 8 experts with weights w_0..w_7, sorted descending
  IMMEDIATE set: top K by weight        -> fetch, compute, fold into layer N output
  DEFERRED set:  remaining 8-K          -> ISSUE THE READS NOW, do not wait
layer N returns as soon as the IMMEDIATE set is done
  the DEFERRED reads are in flight across the whole of layer N+1
layer N+1 collects the deferred partials and adds them to its residual
```

The point is the second line. **The deferred reads must be issued before layer N returns**,
not when layer N+1 starts — issuing late reproduces the gap we are trying to close.

Requirements:

- `WP_DEFER_K` (experts computed immediately, default = all = feature off). Off by default,
  opt-in, and its **actual runtime value printed in the log at startup**. A gate left at a
  permissive default means you measured the ungated system; that has cost this project three
  separate retracted measurement sets.
- Deferral selection is by **router weight**, lowest deferred. The dispatcher already has the
  weights — `pipe-expert-dispatch-graph.cpp` builds the `[1, n_tokens]` weight tensor.
- Deferred partials must carry their `(layer, token)` identity so layer N+1 folds them into
  the right residual. Do not assume ordering.
- **The last routed layer has no successor.** Its deferred set must be folded before the
  final norm, or not deferred at all. Decide explicitly; do not let it fall off the end.
- Prefill and decode differ: prefill has `n_tokens > 1` and its own routing per token. Either
  disable deferral during prefill or handle the per-token bookkeeping. Do not let it silently
  half-work — the 2026-07-30 broadcast bug was exactly a decode-correct/prefill-corrupt split
  that looked fine until it poisoned the KV cache.

## 4. Mechanism counters — the counter is UTILIZATION

```
nvme_util_pct        device busy fraction, from /proc/diskstats io_ticks
ns_gap               time per layer with ZERO reads in flight
n_deferred           experts deferred, cumulative
n_deferred_late      deferred partials that arrived AFTER their fold point (a bug counter)
```

**Do not gate this work on queue depth** — it is already 11-14 and will barely move, so it
would read as "no effect" while the fix works. **Do not gate it on tok/s** — that is an
outcome and it is confounded by miss rate and cache state, which is how a cold-cache artifact
once got shipped as "8.7x".

If `nvme_util_pct` does not move off ~40% (2026) / ~30% (main), the deferral did not happen,
regardless of what throughput says.

## 5. Expected effect, written down in advance

- Utilization is the prediction: **40% -> 60-80%** on 2026 if deferred reads genuinely span
  layer N+1.
- At 100% occupancy the read leg improves ~2.4x on 2026 and ~1.5x on main. That is a
  **ceiling, not a forecast** — partial overlap gets a fraction of it, and read is one leg of
  three.
- No end-to-end tok/s claim is made here. Report per-leg and utilization.
- Steady-state baseline is **~0.99 tok/s over 256 tokens**, not 0.89. Every 32-token figure
  in the older docs is prefill-inflated by ~11%. Compare against 256-token runs only.

## 6. Re-sharding is OFF THE TABLE

Do not propose moving the expert-index split, under any justification. kmbandy has rejected
it repeatedly and it is not to be raised again — not as "capability-weighted re-shard", not
as "load rebalance toward the idler drive", not as any other relabelling of the same idea.
The shard ratio was solved from measured NVMe bandwidth and it stays where it is.

The occupancy gap is to be closed by deferral (this document), and later by prefetch behind
DSpark. Not by moving work between machines.

## 7. Sequencing

kmbandy's call, 2026-07-31: deferral before prefetch, DSpark before prefetch. The reason is
measured, not preference — the 2026-07-10 cross-layer prefetch attempt regressed DS4 from
1.629 to 1.420 tok/s (-12.8%) for lack of lead time, issuing only 23 speculative pages over
128 tokens x 43 layers. A draft model *is* the lead time. Building prefetch before DSpark
rebuilds the thing that already failed.

Note DSpark also lands via the upstream sync (merged upstream as 84075273c, plus #25683 for
V4 targets), which is its own scoped job — see the sync scoping note in the KG.

## 8. Build — every target, both machines

```
mad-lab-main   build-hip    llama  llama-server  llama-wp-expert-worker  test-wp-expert-worker
mad-lab-2026   build-army   llama  llama-server  llama-wp-expert-worker  test-wp-expert-worker  (-j2)
```

On 2026, move the mapped `libllama.so*` / `libggml*.so*` chain aside before rebuilding so the
live services (pid 855466 nemotron embedder, pid 3025042 llama-router) keep their mapped
inode. `mv` within the filesystem preserves the inode for already-running processes; the build
then creates new files at the original names. Never signal or restart those services.
