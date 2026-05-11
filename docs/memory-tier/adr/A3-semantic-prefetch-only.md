# A3. Semantic drives prefetch only, not eviction

**Status**: Accepted (2026-05-10)
**Decided in**: [MAD-126](https://mad-lab-ai.atlassian.net/browse/MAD-126)
**Implemented in**: MAD-129 (semantic prefetch on `llama_kv_cache_paged`)

## Context

A natural extension of having a per-block embedding (BGE-small
fingerprint, see [A2](A2-fingerprint-at-prefill.md)) is to use it for
**eviction** decisions, not just prefetch — pick the block to evict
that's least semantically related to the current query.

This was considered and rejected.

## Decision

bge-small drives **prefetch only**. Eviction stays on the existing
hybrid policy in `src/memory-tier/mt-eviction.{h,cpp}`, which weights
recency + attention magnitude + frequency without any semantic input.

## Reasoning

1. **Training-task mismatch.** BGE-small is trained for retrieval —
   "given a query, find documents that answer it." That's a cousin
   problem to "predict which cached blocks an LLM is about to attend
   to," but only a cousin. There's no published evidence that
   retrieval embeddings correlate well with attention-cache
   predictiveness, and there's no ground-truth dataset to tune on.

2. **Hot-path latency.** Every eviction decision would need an embed
   call, or a lookup against pre-computed embeddings plus a cosine
   computation per candidate. Eviction needs to be fast (<1ms)
   because it sits in the prefill / decode critical path. Adding
   embed lookups inflates the decision cost meaningfully.

3. **Doesn't fix the structural problem.** The thing that actually
   threatens correctness under multi-seq load is hot-pool
   fragmentation when N agents collectively want more hot blocks
   than exist (MAD-120's admission control problem). Smarter
   eviction picks doesn't change that — admission control does.

4. **Measurable risk if wrong.** If semantic-driven eviction
   picks badly, decode quality degrades silently (the model attends
   to misleading old context with wrong weights). Whereas
   recency/frequency-driven eviction has known failure modes that
   produce predictable degradation.

## Consequences

**Positive**:
- Eviction decisions stay simple, fast, and analyzable. The hybrid
  policy in `mt-eviction.cpp` is ~200 lines of boring math.
- The semantic path is purely **additive** — turning it off (no
  `--kv-tier-semantic-index`) leaves the cache fully functional.

**Negative**:
- Some workloads where semantic-driven eviction would have been a
  win don't get that win. Specifically: long-running agent
  workflows where the "right" thing to evict is defined by what the
  CURRENT query semantically isn't asking about. Re-evaluate this
  decision if such workloads become important.

**Neutral**:
- Two separate concerns now own two separate code paths: eviction in
  `mt-eviction`, prefetch in `BlockSemanticIndex` +
  `llama_kv_cache_paged::restore_semantic_paged`. Easier to reason
  about, but anyone wanting to introduce semantic-driven eviction
  later has a clear refactor surface.
