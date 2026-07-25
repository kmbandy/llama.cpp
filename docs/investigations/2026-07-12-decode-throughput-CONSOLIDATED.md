# Decode throughput — consolidated findings

**Date:** 2026-07-12
**Inputs:** two independent investigations, both run blind, neither given a hypothesis, neither shown
the other's work. Both ran on the same card (RX 6900 XT, ROCm1). Both were told to distrust
`GPU use %` and to execute rather than read.

- `2026-07-12-decode-throughput-findings-fable.md`
- `2026-07-12-decode-throughput-findings-codex.md`

---

## Both agree, independently (all `executed`)

| finding | fable | codex |
|---|---|---|
| plain llama.cpp single-stream decode | **115 tok/s** | **112.6 tok/s** |
| plain decode scales with concurrency | 112 → 282 → **380** (N=1/8/18) | 106.8 → 244.8 → **322.6** (N=1/8/18) |
| the model, the GPU, and concurrency are all **fine** | yes | yes |
| the wall is **our own KV stack** | yes | yes |
| `GPU use %` is meaningless; power is the signal | yes | yes |
| turbo4 costs something but is **not** the wall | ~2x | ~34% |

**Concurrency was never broken. Decode was never slow. Every decode-shaped theory — including my
own split-K thesis — was aimed at a component that was working correctly.**

---

## THE ROOT CAUSE (codex — this is the sharpest result of the day)

Codex did the isolation Fable structurally could not: it **separated tiering from paging.** Fable's
ladder collapsed them into one step. Codex changed one flag at a time:

```
plain,                       N=18:  322.6 tok/s
+ turbo4,                    N=18:  212.7 tok/s      (-34%)
+ tiering 75/25, paging OFF, N=18:  228.5 tok/s      <- FREE. Tiering costs NOTHING.
+ paged blocks ON,           N=18:   61.5 tok/s      <- 3.7x collapse
+ fleet capacity 1.5M ctx,   N=18:   21.5 tok/s      <- another 2.9x (same active depth!)
+ active depth ~10k,         N=18:   10.4 tok/s
```

Final matched A/B, **only** `--kv-tier-paged-blocks` changed, everything else held (turbo4, tiering,
full 1.5M capacity, N=18, exact 10k history, `prompt_n=1`):

```
paged ON :  10.448 tok/s,  92.1 W
paged OFF: 140.263 tok/s, 172.4 W
                                     13.4x
```

**Tiering is free. Paging is the wall.** And the N=1 control stays healthy at ~100 tok/s — so it is a
**concurrency-dependent** regression, not a constant tax.

### The mechanism (codex, `source` + inference — NOT yet profiler-confirmed)

`ggml/src/ggml-cuda/mt_pagedattn.cu:1760-1799` — the custom flash-decode dispatch only accepts:

```
total_q_tokens <= 8   AND   num_queries_per_kv * total_q_tokens <= 16
```

**A normal N=18 decode step has 18 query tokens. It misses the gate.** It then falls through to the
**scalar fallback** (`:1840-1875`), which walks the active context in 256-token chunks, **token by
token**, for QK and V accumulation (`:1406-1512`).

This single mechanism predicts every symptom we could not explain:

| symptom | explained by the scalar fallback |
|---|---|
| N=1 fast, N=18 collapses | 1 query token passes the gate; 18 does not |
| degrades with context depth | the scalar kernel *walks* the context |
| low power at "100% busy" | scalar kernel: no WMMA, minimal occupancy |
| prefill also degraded | same paged path |

There is already a comment at `mt_pagedattn.cu:1601-1606` recording a **prior** regression
investigation of exactly this collapse. Codex found it only *after* the A/B isolated paging.

**This is also why split-K was the wrong target and the right instinct.** The scalar fallback *is* an
under-parallelized decode kernel — but the fix is not to write a better kernel. It is to **stop
falling off the dispatch gate into it.**

---

## THE SECOND BUG (fable — real, independent, and NOT tested by codex)

Codex's ladder **never enabled `--kv-tier-semantic-index`.** Its recommendation "the first target is
the gate, not the semantic embedder" is therefore `inferred`, not `executed` — it never measured the
embedder. Fable did, and holding context depth pinned at 5,745 tokens in every config:

```
+ semantic index (on top of paged):  5.04 -> 1.49 t/s per stream    5.35x
```

**Mechanism (`source`):** `server-context.cpp:4136` calls `embed_text_batch()` **synchronously inside
`update_slots()`** — the one loop that advances every slot. The sweep runs **~40 s per slot,
serially**; 24 slots ≈ 16 minutes of a 17-minute run, and slots already mid-generation are **frozen**
for each 40 s window.

Two aggravating defects:
- `mt-embed.cpp` sets `n_gpu_layers = 0` and comments "CPU only" but **never sets
  `mparams.devices`** — so the embedder still builds a GPU backend and schedules ops on it. It is
  competing with the model it exists to serve. (Same defect that OOM'd the RX 480's child.)
- **Widening the embedder's context pool cannot help** — the caller *blocks*. (A pool was built and
  A/B'd: 4.8 → 5.4 t/s, noise. That result was then wrongly generalized to "the embedder is
  innocent.")

**Both bugs are real and they are independent.** Paging is the bigger term; the embedder sits on top
of it.

---

## What is NOT the problem (measured, do not re-chase)

- **The model, the GPU, concurrency** — all fine. Plain decode scales.
- **Tiering (75/25 split)** — costs nothing without paging. `executed`, codex.
- **turbo4** — a real cost (34%–2x) but nowhere near the wall. Do not remove it: it is what makes the
  context budget fit at all.
- **The idle "99% busy / 79 W" spin** — costs nothing (111.9 vs 114.9 t/s). `executed`, fable.
- **Context over-provisioning** — REJECTED. Fable proposed cutting ctx to 16k/slot on the premise
  that "contexts are 3k–10k." The real run had a cell at **60,149 tokens** that produced a good
  result. Cutting context would amputate the reasoning depth, not fix the bug. The 65,536/slot budget
  is earning its keep.

---

## Corrections to my own arithmetic (codex, `derived`)

- KV/token: **4.84–5.20 KiB/token** depending on GB-vs-GiB. My "~4.75 KB/token" was the right order
  but not a clean derivation.
- My "10 GFLOP/s useful decode compute" omits attention over context, routing, recurrent/conv work,
  and dequant. **It is not a valid utilization estimate for this failure.** I overclaimed.

---

## What to do

Ordered by (measured payoff) / (risk). **Nothing here disables a feature.** The tiered/paged/semantic
stack is the differentiator — it is why 8 agents run at 64k on decade-old 8 GB cards, and it is what
anyone else running this fork on a small GPU depends on. These are **bugs in our moat, not a price we
pay for it.**

1. **Fix the paged decode dispatch gate.** `mt_pagedattn.cu:1760-1799`. A multi-sequence decode step
   with N sequences has N query tokens and must not fall off a gate sized for `<= 8`. Confirm the
   branch first with `rocprof` or `MAD_PAGEDATTN_PROBE=verbose` — codex rates the mechanism `medium`
   confidence and explicitly says to confirm before changing dispatch. Expected: up to **13.4x**.

2. **Get the fingerprint sweep off `update_slots()`.** A background worker with its own
   `llama_context`; enqueue-and-continue. Fingerprints are metadata — nothing in decode needs them
   synchronously. **Also pin the embedder to CPU** (`mparams.devices`), or the worker just moves the
   GPU contention rather than removing it. Expected: up to **5.35x**.

3. **Investigate why configured capacity is itself a throughput variable.** Codex: raising total ctx
   5.33x cut throughput another 2.9x **at identical active depth**. That means paged cost scales with
   *configured* capacity, not *used* tokens — which is a bug shape, not a design cost. Note the server
   prints `n_ctx_slot = 1572864` under tiering vs `16384` in the plain control: the tiered path gives
   each slot the **full** context, which is a semantic difference worth understanding.

4. **Do NOT ship `--no-kv-tier-paged-blocks` as the answer.** It is a valid diagnostic and a valid
   emergency mitigation. It is not a fix — it turns off the feature.

## Open, and worth measuring before building

- **Do fingerprints survive across agentic turns?** Keyed by `slot.id + logical block`. Fable measured
  a single turn (all blocks new). If a slot is reassigned to another conversation, or a context shift
  renumbers blocks, the cache invalidates and **the full sweep recurs every turn** — making production
  far worse than the measured 5.35x. Cheapest high-value test available. `inferred`.
- **What does the semantic index actually buy?** We have now measured its cost to three significant
  figures and its *retrieval benefit* not at all.
- **Is the turbo4 cost inherent?** 4-bit KV reads 1/4 the bytes of f16 and should make deep-context
  decode *faster*. It is 34%–2x slower. That asymmetry deserves the same treatment.

---

## Method note

Two investigators, same problem, same card, no shared hypothesis, both required to execute rather
than read. They converged on the diagnosis (our KV stack), diverged on the component, and **the
divergence was the point**: Codex separated tiering from paging and found the gate; Fable measured the
embedder Codex never enabled. Either one alone would have produced a confident, incomplete answer.

The main-loop investigator (me) produced four confident wrong root causes in one day and had to be
corrected by measurement each time. The single highest-leverage thing done today was refusing to tell
either investigator what we thought the answer was.
