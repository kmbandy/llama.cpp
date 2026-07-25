# Parallel decode throughput for the synth-data fleet — RESULTS

**Measured 2026-07-14 by claude__main, answering
`2026-07-14-parallel-decode-synth-fleet-BRIEF.md`.**

## The finding

The fleet's aggregate throughput was collapsing because **the semantic index runs a
synchronous CPU embedder forward pass on the server's single-threaded inference loop.**
While that pass runs, *no slot decodes*. The cost therefore scales with slot count —
which is exactly why adding parallelism bought nothing.

This is MAD-348. It is now **fixed**, not worked around. With every core feature still
enabled (semantic index + KV tiering + paged blocks), the 6900XT goes

> **100.2 → 212.3 tok/s at 24 streams (2.1x)** — 93% of what the same config achieves
> with the semantic index *removed entirely* (228.0).

An earlier draft of this document recommended turning the KV stack off for the synth job.
That was rejected, correctly: the features are needed for other workloads, and "don't use
the feature" is a symptom fix. The measurements below are the same, but the conclusion is
a code fix.

## Root cause (instrumented, not inferred)

The brief's rule — *two decode fixes were built on plausible reasoning and both measured
exactly zero; get timing before attempt #3* — was followed. Instrumenting
`EmbeddingModel::embed_batch` under load on the real murmur config gave:

```
160 calls, 45.7 s cumulative in embed_batch (avg 285.4 ms/call), max in-flight = 1
```

Two facts, both decisive:

1. **~40–60% of the server's wall-clock was spent inside `embed_batch`.** (Between the
   last two probe lines: 10.3 s of embed inside 16.8 s of wall time = 61%.) The caller is
   `update_slots()` — single-threaded — so every one of those microseconds is a
   microsecond in which no slot decodes.

2. **`max in-flight = 1`, always.** `EmbeddingModel` maintains a pool of `parallel_`
   contexts and correctly drops the class mutex across `llama_decode`. But the pool could
   never be used, because **the only caller was the single server loop**.
   `--kv-tier-semantic-parallel 3` allocated three contexts that never ran concurrently.
   The lock had been fixed; the *caller* had not. The comment in `mt-embed.cpp` claiming
   "with parallel_ > 1 the slots embed concurrently instead" was simply false.

Per synth request the server paid **two** embeds: a query embed (for semantic restore) and
a 7-block fingerprint sweep — block size is 16, so a 124-token prompt produces 7 complete
blocks, and with every synth prompt unique, all 7 were new every time.

### Why the embed cost 285 ms

`mt-embed.cpp` carried this comment:

> *"the model is tiny (~30 MiB for bge-small) and contention with the main model's VRAM
> isn't worth saving a few ms of inference latency."*

Both halves are stale. The fleet no longer runs bge-small — it runs
**LFM2.5-Embedding-350M-Q8_0**, ~10x larger — and it is not "a few ms", it is 285 ms on
the critical path. The embedder model was swapped and the CPU-only decision was never
revisited.

## The fix — three changes

**1. Don't embed for a search that cannot return anything.**
`restore_semantic_paged()` can only act on a fingerprinted block that is mapped and *not
already hot*. If every block is resident, it returns 0 regardless of the query vector —
but the server paid the full 285 ms embed *before* discovering that. On a short-context
job nothing is ever evicted, so this fired on every request and restored nothing.
New predicate `llama_kv_cache_paged::has_restorable_blocks()` (a handful of pointer
checks) gates the embed.
→ `src/llama-kv-cache-paged.h`, `tools/server/server-context.cpp`

**2. Run the fingerprint sweep off the inference thread.**
The sweep is a pure *write* — nothing later in the request reads the fingerprints back,
so there is no read-after-write dependency to honour. It now dispatches to a background
worker via `std::async`, and is joined in `slot.release()`.
**That join is load-bearing:** a sweep outliving its request would call
`record_paged_block_fingerprint()` with a slot id that a *later* request had already
taken over, writing a fingerprint that describes the old content but is keyed to the new
sequence — silently poisoning the index. Joining at the reuse boundary makes that
impossible.
→ `tools/server/server-context.cpp`

**3. Pin the embedder to a CPU-only device list.** ← *prerequisite for #2*
`n_gpu_layers = 0` **is not enough**. With `mparams.devices` unset, `llama_init_from_model`
still builds a GPU backend for the embedder and schedules ops on it. This is not
speculation — it caused two separate, already-documented failures:

- On mad-lab-2026, the RX 480's scout died with `CUDA error: out of memory` because its
  stray CUDA context landed on **CUDA0 (the 1070)**, already full. The workaround was to
  strip `--kv-tier-semantic-index` from the 480 entirely, **giving up eviction quality**.
  The 2026 preset says in a comment: *"Restore this line once mt-embed.cpp pins the
  embedder to a CPU-only device list."* That is now done — **the 480 can have its
  semantic index back.**
- Change #2 initially **crashed the server**: `ROCm error: operation would make the legacy
  stream depend on a capturing blocking stream`. The background embed's GPU ops raced the
  main model's HIP graph capture. Async didn't cause this — it *exposed* it.

→ `src/memory-tier/mt-embed.cpp` (+ `std::call_once` on the lazy embedder init in
`mt-tiered.{h,cpp}`, which becomes a data race once the caller is multi-threaded)

## Verification

- **Correctness first.** The async change's first cut produced **157 HTTP 500s**. It was
  not shipped; it was root-caused (see #3). The final build runs clean: **0 failures, 0
  ROCm errors, 0 aborts** across every run.
- **The feature still works.** 196 fingerprint sweeps dispatched under load; restores
  still fire.
- **New regression test** for the guard's safety property in `tests/test-paged-semantic.cpp`.
  The property is one-directional and absolute:
  `restore_semantic_paged() can restore > 0  ⟹  has_restorable_blocks()`.
  A false negative would silently disable semantic prefetch — a quality regression no
  throughput number would ever reveal. The test evicts a fingerprinted block to warm and
  asserts the guard opens *and* that the permitted restore actually restores.
- **Full tiered suite passes**: `test-paged-semantic`, `test-memory-tier`,
  `test-mt-block-semantic-index`, `test-mt-tiered-thin`, `test-paged-lifecycle`.
  ⚠️ These tests **silently SKIP** without `LLAMACPP_TEST_MODELFILE` set. They were run
  with `LLAMACPP_TEST_MODELFILE=~/models/LFM2.5-Embedding-350M-Q8_0.gguf`. A green run
  without that env var means nothing.
- `--kv-tier-semantic-parallel` is now genuinely live (it was inert before). Widening it
  3 → 8 changes nothing (212 → 208, noise), because the embed is no longer on the critical
  path at all. **Leave it at 3.**

## Numbers (6900XT, LFM2.5-8B-A1B Q8_0, short prompt / 128 new tokens)

| concurrency | murmur BEFORE | murmur AFTER (all features on) | no-semantic reference |
|---|---|---|---|
| 1 | 48.8 | 59.2 | 71.5 |
| 8 | 56.7 | 78.2 | 108.7 |
| 24 | 100.2 | **212.3** | 228.0 |

## The second finding: drive the slots full

Independent of MAD-348, and a **client-side** bug that no server fix can reach:

**Throughput tracks full slot occupancy.** At `--parallel 64`, driving only 24 concurrent
streams yields **147** tok/s; driving 64 yields **393** (three reps: 406 / 391 / 381 —
reproducible, not the clock-decay confound). Under-filling slots wastes the card.

**The synth harness must keep exactly `--parallel` requests in flight at all times.** This
is worth ~2.7x on its own.

Saturation on the 6900XT is at 64 slots: `--parallel 128` gives 388.5 tok/s — same
throughput, double the latency (31 s vs 17 s), more VRAM. 64 is the knee.

## Fleet aggregate

**554.8 tok/s across three GPUs simultaneously** (60 s steady-state window):

| GPU | `--parallel` | drive at | tok/s |
|---|---|---|---|
| RX 6900 XT | 64 | 64 | 400.0 |
| GTX 1070 | 16 | 16 | 92.7 |
| RX 480 | 8 | 8 | 62.1 |
| | | **total** | **554.8** |

**GPUs do not contend.** The 6900XT did 393 solo and 400 under full fleet load; the 1070
did 92.6 solo and 92.7 in fleet. Scaling across cards is linear — whatever was wrong, it
was never cross-GPU interference.

The **R9700 was excluded by instruction** and is not in this number. It is the fastest card
in the fleet; adding it should push well past 900 tok/s, but that is an extrapolation.

## Null results (worth as much as the wins)

- **Decode was never broken.** Single-stream 101 tok/s and 317 at 24 streams land right on
  the brief's stock ceiling (112–115 / 322 at N=18). Every decode-shaped theory would have
  been aimed at a component that works — exactly as the brief predicted.
- **The paged-attention dispatch gate (brief item 4) is not implicated.** Not probed,
  because the profile put the time in `embed_batch`, not in attention. Still worth probing
  if the scouts are ever tuned.
- **The RX 480 is compute-bound and doesn't care.** Synth at par8 = 68.6 tok/s; murmur at
  par8 = 76.3. The KV stack costs it ~nothing — it is slow enough that decode compute
  dominates. `--parallel 16` (54) actually LOST to `--parallel 8` (76). **"More slots" is
  a big-card rule.** Don't tune the 480.

## Follow-ups

1. **Restore `kv-tier-semantic-index` on the RX 480** in `router-fleet-2026.ini`. The
   CPU-device pin (#3) removes the OOM that forced it out. Verify before trusting.
2. The query embed still blocks *its own* slot when a restore is genuinely possible. That
   is correct-but-serial; if deep-context scouts ever show the same stall, the slot state
   machine needs an `AWAITING_SEMANTIC` state so the loop can serve other slots meanwhile.
3. Ablation aliases in the presets (`*-ablate-*`, `*-synth*`, `*-sp8`) exist only to
   attribute these numbers. Delete when done.
