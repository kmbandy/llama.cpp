# Parallel decode throughput for the synth-data fleet — investigation brief

**Written 2026-07-14 by claude__main. This is OBSERVATIONS ONLY — deliberately.**

There is no diagnosis in this document and that is on purpose. On 2026-07-12 the single
highest-leverage act of that investigation was refusing to tell the investigators what we
thought was wrong; two independent sessions then converged on the diagnosis and *diverged*
on the component, and the divergence was the point. Every decode-shaped theory we held
going in was aimed at a component that turned out to work.

So: here is what we want, here is what has been measured, here is what is unknown. Go find
out. Do not trust the framing below to be complete.

---

## What we need

Generate synthetic `memory`-domain training data for MAD-160 (mlambaformer). The corpus
audit on 2026-07-14 found the memory domain has ~8M real tokens against a 250M budget, so
the volume has to be synthesized.

**The job shape:**

- Model: **LFM2.5-8B-A1B** (`~/models/LFM2.5-8B-A1B-Q6_K.gguf`)
- **Low context**, no reasoning budget. Feed it a format + a few real examples + a random
  subject; it emits one synthetic memory record.
- **Many short independent generations.** Embarrassingly parallel — no shared state.
- Several instances **across all 4 GPUs**.
- The metric that matters is **aggregate tokens/sec across the whole fleet**, not
  single-stream latency. Nobody has ever measured that number for this shape.

## What is measured (trust these; they are numbers, not theories)

From the 2026-07-12 decode investigation (docs/investigations/2026-07-12-decode-throughput-*):

- **Stock decode is fast and scales with concurrency.** 112–115 tok/s single-stream, flat
  with context depth. Concurrency *helps*: **106 → 244 → 322 tok/s at N=1/8/18.**
- **Prefill's real number is 5100–6170 tok/s**, not the 335 we had been misreading as
  healthy.
- **`--kv-tier-paged-blocks` ON vs OFF: 10.4 → 140.3 tok/s (13.4x).** Matched A/B, that
  flag the only variable. Power 92.1 → 172.4 W.
- **Tiering itself is free** (228.5 vs 212.7 tok/s = noise). Paging and tiering are
  separable and were separated.
- **MAD-348: the semantic index blocks `update_slots()` synchronously.** ~40s per slot,
  serially, slots mid-generation frozen. 5.35x, measured with depth pinned.

From MAD-301 (the RX480/gfx803 army, which already serves this exact model):

- LFM2.5-8B-A1B decode ~59 tok/s on gfx803 after CHUNK_KV 256→128.
- Prefill 8.3x via an MMQ routing fix (gfx803 has no hardware dp4a).

Model facts, read from the GGUF header 2026-07-14:

```
n_heads = 32,  n_kv_heads = 8   (GQA fanout 4)
24 blocks, of which only 6 are attention — head_count_kv is a PER-LAYER array
[0,0,8,0,0,0,8,0,0,0,8,0,0,0,8,0,0,0,8,0,0,8,0,0]  — the rest are conv.
context_length 128000, 32 experts, 4 active.
```

## What is unknown

1. **What is the max aggregate decode throughput for this workload across the 4 GPUs?**
   Nobody has measured it. Everything above is single-server, and mostly on other models.
2. **What is the right `--parallel` per instance, and how many instances per GPU?** We run
   `--parallel 8` on the smaller GPUs today and it works. Whether 8 is optimal for *this*
   model at *low* context is unmeasured.
3. **Which flags should the synth fleet actually run with?** The dashboard launcher
   (`mad-lab-dash/mad-dashboard.py:444`) does not pass any `--kv-tier-*` flag; other launch
   paths on mad-lab-2026 are not in this checkout and were not read. **Find out what the
   fleet actually runs before assuming anything.**
4. **Is there a dispatch gate that this workload falls off?** There is a gate at
   `ggml/src/ggml-cuda/mt_pagedattn.cu:1795` whose conditions involve `total_q_tokens`, the
   GQA fanout, and `max_ctx_len >= 8192`; when it fails, decode goes to a scalar kernel
   (`:1840-1875`) that walks the context token-by-token. A low-context, high-parallelism
   job is an unusual shape for it. **Whether this is live on our launch path is EXACTLY the
   kind of thing to confirm with a probe rather than by reading code — see below.**

## Discipline notes, paid for in real time

- **Confirm the branch before touching dispatch.** `MAD_PAGEDATTN_PROBE=verbose` prints
  which path each call takes (`mt_pagedattn.cu:1800-1806`). The 2026-07-12 mechanism was
  source-inferred and never profiler-confirmed.
- **Two decode fixes have already been built on plausible reasoning and both measured
  exactly zero** (MAD-301A attempt #2: 34.4 vs 34.1 tok/s, reverted). Get per-kernel timing
  before attempt #3.
- **Split-K decode has been proposed twice and was wrong both times.** It may still be
  right! But it has a track record and it needs evidence, not an argument.
- **Distrust GPU-use%.** It reads 99–100% at 67W of a 300W budget.
- **Pin the confound:** generation opens ~3x high and decays over 2–3 minutes. Compare
  like-for-like or you will fool yourself. (We did. Twice.)
- **`--split-mode row` for multi-GPU.** The default (`layer`) pipelines activations through
  the CPU between GPU stages.

## The deliverable

A configuration — instances per GPU, `--parallel`, context size, flags — and the
**measured aggregate tok/s** it achieves for LFM2.5-8B-A1B on short-context, high-fan-out
generation. If a code fix is needed to get there, that is a finding; if it turns out no fix
is needed, **that is equally a finding** and is worth just as much.
