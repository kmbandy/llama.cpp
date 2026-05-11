# A8. bge-small ownership stays on `mt::llama_memory_tiered`

**Status**: Accepted (2026-05-10)
**Decided in**: [MAD-126](https://mad-lab-ai.atlassian.net/browse/MAD-126)
**Implemented in**: MAD-127 (wrapper thinning) + MAD-129 (server-side bridge)

## Context

After [A1](A1-hybrid-paged-primary.md), `mt::llama_memory_tiered` is a
thin shim — most of its old responsibilities moved into
`llama_kv_cache_paged`. Two responsibilities remained candidates for
the wrapper vs. the cache:

1. **bge-small embed model ownership.** Loading the gguf, holding
   the `EmbeddingModel` instance, exposing `embed_text(string) →
   vector<float>` to whoever needs to compute a fingerprint.

2. **Recurrent-state backup on hybrid models.** Backing up the
   per-seq recurrent state into host RAM on `seq_rm` so it can be
   restored later (the recurrent half's `clear()` is destructive).

Putting these on `llama_kv_cache_paged` would have made the cache the
single owner of all tier-related state. But it would also have
spread embed-model concerns into the cache header and forced the
cache to know about recurrent state — neither is its job.

## Decision

Both responsibilities stay on `mt::llama_memory_tiered`:

- `mt::llama_memory_tiered::embed_text(text)` is the embed entry point.
  Lazy-loads the gguf on first call (or warmup at construction —
  see MAD-134). Returns L2-normalized 384-dim vectors.
- `mt::llama_memory_tiered::backup_seq_rm_recurrent` /
  `restore_recurrent_from_warm` are the recurrent backup pair.

The paged cache holds the resulting fingerprints in its own
`BlockSemanticIndex`. The server bridges the two via the existing
`mt_get_paged_cache(llama_get_memory(ctx))` accessor: it asks the
wrapper to compute an embedding, then hands the embedding to the
cache's `record_paged_block_fingerprint`.

## Consequences

**Positive**:
- The cache stays focused on its core responsibility (block
  management, eviction, K/V tier movement). It doesn't know what an
  embedding model is.
- The bge-small instance is shared across configurations — pure-
  paged, hybrid+paged, and the legacy thin-wrapper paths can all
  use it without each loading their own copy.
- Recurrent-state backup is a hybrid-specific concern; keeping it on
  the wrapper means a pure-attention config with paged routing
  doesn't carry the recurrent-backup code paths at all.

**Negative**:
- Adds an indirection: server → wrapper.embed_text → cache.record.
  Extra function call per fingerprint, negligible cost.
- The wrapper survives in a "thin but not gone" state, which is a
  slight wart if you wanted A1 to fully kill it. Acceptable
  tradeoff given the cleanliness benefit of keeping these two
  concerns separate from the cache.

**Neutral**:
- Anyone wanting to swap bge-small for a different embedding model
  changes one place (the wrapper's `embed_text`). The cache and the
  fingerprint store don't care which model produced the embedding,
  as long as it's L2-normalized 384-dim.
