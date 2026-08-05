# Measurement discipline for the DS4 distributed-MoE stack

**Written 2026-08-05, after a session in which the same mistake occurred five times.**

This is not general advice. Every rule below is here because it was violated on this
codebase, and each one names the instance that motivated it. Read the instances —
they are the part that transfers.

---

## The core failure mode

> **A number was quoted without first checking what was inside it.**

Not "the measurement was noisy." Not "the hypothesis was wrong." In each case the
arithmetic was correct and the counter was read correctly — and the *contents* of the
counter were not what the name suggested.

Five instances in one session:

| what was quoted | what it actually contained |
|---|---|
| `ns_compute` as GPU time | a **superset** — it is stamped before `prepare_io` and closed after `batch.complete()`, so it contains `ns_read` and `ns_h2d`. There was already a retraction on file for exactly this. |
| "verify costs 8.2 ms/tok" | a median dominated by **183 cheap 3-layer NextN blocks**. The real 43-layer main verify is **361.7 ms**. |
| "33.5 MB payload per request" | the **ubatch cap** (2048), not the real prefill width (659 tokens = 10.8 MB). A 3.1× error that propagated into sizing a code change. |
| "27.6% overlap deficit" | Σ-of-per-layer-maxima compared against max-of-per-layer-sums. See rule 2. |
| `bare_n1_blocks` | a count whose **label is assigned by the swept variable**. See rule 1. |

---

## Rule 1 — Does the metric move with the manipulation for reasons other than the effect?

The dispatch block log assigns its phase label purely by width:

```
n_tokens >= 64 -> PREFILL      n_tokens > 1 -> verify      n_tokens == 1 -> decode
```

A sweep of `SPEC_NMAX` changes the **draft block width**. At `nmax=1` the 3-layer
NextN draft blocks become `n_tokens=1` and are relabelled "decode". So the count of
"bare decode blocks" rose from 88 to 290 — and none of that was the effect under test.

**A label assigned by the swept variable can never measure that variable's effect.**

Before any sweep, ask what *else* the swept variable touches in the measurement path.
Write the answer down. If the metric is downstream of the manipulation by any route
other than the mechanism, pick a different metric.

## Rule 2 — Sum-of-maxima is not maximum-of-sums

```
wait        = SUM over layers of  max over workers  (per-layer straggler)
max(worker) = max over workers of SUM over layers   (worst worker overall)
```

`Σ max ≥ max Σ` **always**, and the gap grows whenever the straggler *alternates*
between layers — which near-equal workers guarantee. The per-block log line sums 43
layers, so this is invisible unless you look for it.

This produced a phantom "27.6% serialisation deficit" and a task (#31) that was
already refuted by a comment sitting three lines above the flag it concerned.

Any time two aggregates are compared and one is an inner-max and the other an
inner-sum, the comparison is invalid.

## Rule 3 — Read the raw rows before quoting any summary

The one analysis that was right first time used `WP_REQ_LOG` and segmented
**explicitly** by `n_tokens`, reporting counts alongside values:

```
n_tok  count   n_assign  submit_ms  submit/tok  us/tok/expert
  1     3648      1        0.310      0.3099       309.9
  2     2555      2        0.458      0.2288       114.4
  4       42      3        0.585      0.1463        48.8
659       43     31       25.609      0.0389         1.3
```

That single table answered a question two hypotheses had failed to answer, and it
also revealed the headline finding of the session (n=2 costs the same as n=1).

**Always print the sample count next to the value.** A bucket with n=1 and a bucket
with n=3648 must not be read the same way.

## Rule 4 — Read the code around a lever before proposing it

`WP_DISPATCH_HARVEST` already existed, and the comment directly above the flag
recorded that it had been built, measured, and had *not* recovered the time. A task
was filed to build it anyway.

The prior RX 480 investigation had already refuted six hypotheses (arch
misclassification, the mmvq path, per-expert buffers, clock ramp-down, pool size,
H2D contention) and named the diagnostic that would settle it. That diagnostic
(`GGML_VK_PERF_LOGGER`) had never been run, and running it took twenty minutes and
produced the session's most important number.

**Grep the flag name and read every comment near it before writing anything.**

## Rule 5 — A passing test proves nothing until you know it can fail

`test-wp-expert-worker` verified returned partials with:

```cpp
ggml_fp16_to_fp32((ggml_fp16_t) response.partial[i])
```

`partial` became `std::vector<float>` at `PIPE_VERSION 2`, and `ggml_fp16_t` is
`uint16_t` — so the float was truncated to an integer first, and `0.35f` read back as
**exactly `0.0f`**. The assertion could only pass while every expected value sat
within tolerance of zero. It had been vacuous for a day.

A type change silently disarmed the guard, and the suite stayed green throughout.

Related and already on file: an equivalence probe proves only that **two paths
agree**. `WP_SELFCHECK` compared gather against dense while both were unclamped and
both were fed f16 — it reported "ok" through a defect on every layer. Ask what the
probe *cannot* see before trusting a clean result.

## Rule 6 — Establish the noise floor before claiming a win

The decode envelope over 6 reps is **2.54–2.88 tok/s, a 12.3% spread**. Several
claims of "+9.3%" and "+13%" were made before that envelope existed, and all of them
sat inside it.

Standing gate for any throughput claim on this stack:

- discard a warmup arm (there is a real cold-arm-1 penalty)
- **≥6 reps**
- report **min/max**, not mean±sd — decode is not reliably unimodal at n=6, and
  "bimodal" was itself an over-claim at that sample size
- sample and print machine load per rep, so contamination is visible *in the data*

Corollary: **acceptance and mean accepted length are deterministic per config** (all
six baseline reps returned exactly `0.75000` / `1.76`). Those reproduce and can be
read from a single rep. Throughput cannot.

## Rule 7 — Do not compare a ratio across settings of its own denominator

`acceptance = accepted / generated`, and the confidence gate changes `generated`. So
acceptance is not comparable across `conf_min` settings, and `mean_len` is
`1 + accepted/verifications`, which describes only the steps that actually verified.

Separately: acceptance was compared between two arms **that generated different
text** and reported as causal. Passage predictability drives acceptance, so arms with
different SHAs cannot be ranked by it. State the SHAs whenever quoting either.

---

## Instrument reference — what these counters actually contain

| counter | contains |
|---|---|
| `ns_compute` | **superset**: includes `ns_read` and `ns_h2d`. Not GPU time. |
| `ns_send` (worker) | includes **encode** — the clock starts before `pipe_encode_expert_partial`. |
| `stats.ns_issue` (spine) | includes `plan_requests`, assignment copies, allocation and encoding. **Not socket time.** |
| `phase=` label | derived purely from `n_tokens`. Mixes 43-layer main blocks with 3-layer NextN blocks. Split on `layers=` too. |
| `n_weight_nonzero` | token-expert **pairs**, not the token union. Overcounts a token hitting several of one worker's experts. |
| `submit_hist_us` | the `>=8k` bucket is dominated by **prefill requests**, which legitimately do ~600× the work of a decode request. Not an anomaly. |
| `current_link_width` (sysfs) | the device's **local** link. For anything behind a bridge chain (TB3 eGPU) walk upstream — the constraint is the narrowest hop. |

## Available instruments (all default-off)

`WP_REQ_LOG` (per-request phases, self-segmenting by layer wrap) · `WP_REF_LOG`
(policy-independent reference stream) · `WP_PAGEIN_LOG` · `WP_DISPATCH_REQ_LOG` ·
`WP_DISPATCH_UNION` · `WP_SELFCHECK` · `WP_SPINE_STATS=2` · `WP_WARMUP` ·
`WP_KEEPALIVE_US` · `VKEXTRA` (passes `GGML_VK_*` to **Vulkan workers only**, keeping
the CUDA worker as a clean control in the same run) · `GGML_VK_PERF_LOGGER`.

Prefer an existing instrument to a new hypothesis.

---

## Operational footguns

- **Never dispatch a Codex/Kimi handoff to a machine under measurement**, and always
  pass the standing do-not-run list into the handoff. A GitNexus refresh on
  mad-lab-main mid-benchmark took load to 22.01 and cost a full 6-rep arm.
- **~7 is the harness's own during-run load** on mad-lab-main. Do not use a post-run
  `uptime` as the during-run reference — doing so caused a *valid* 6-rep set to be
  discarded as contaminated.
- **A killed arm leaves the spine/worker ports bound.** Wait for the drain or kill by
  explicit PID (SIGINT then SIGKILL; SIGINT alone was ignored for 10 s). Eight sweep
  points were burned launching into bound ports.
- **`:-` substitutes on unset OR empty.** Use `-` for harness defaults so `VAR=`
  explicitly means "disabled".
- **Never `pgrep -f` a pattern your own checking command contains.** A liveness check
  that self-matches never returns false; this has now cost time twice.
- **Sweep at a resolution that can show the shape.** A 4-point sweep with a 0.5-wide
  gap missed a discontinuity at `conf_min=0` (the gate is *disabled* at 0, not
  thresholded) and never tested above the default at all.
