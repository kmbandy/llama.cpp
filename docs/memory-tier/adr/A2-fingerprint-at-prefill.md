# A2. Semantic fingerprint write trigger = prefill time

**Status**: Accepted (2026-05-10)
**Decided in**: [MAD-126](https://mad-lab-ai.atlassian.net/browse/MAD-126)
**Implemented in**: MAD-129 (server-side prefill trigger)
**Supersedes**: MAD-122 (eviction-time trigger; wrong layer)

## Context

The semantic prefetch system needs a per-block BGE-small embedding
("fingerprint") so that future queries can score similarity against
old blocks and pull the right ones back from warm/cold to hot.

The original MAD-122 design wrote fingerprints at **eviction time** —
when a block was about to be moved out of hot, the embedding was
computed from the block's text and stored.

This worked for `mt::llama_memory_tiered` (the legacy non-paged
tiered cache) where eviction was visible at the wrapper level. It
**did not** work for hybrid+paged because:

1. Eviction in `llama_kv_cache_paged` is internal — the server doesn't
   see eviction events. They happen inside `ensure_blocks_for` /
   `evict_lru_to_warm` / MAD-120 admission.
2. The server-side proactive backup gating doesn't fire for
   hybrid+paged either — its cap arithmetic uses full ctx since
   `physical_attn_cells()` returns 0.
3. Even if both above were fixed, eviction-time fingerprinting writes
   *after* the data is already mid-evict — racy and order-dependent.

A smoke test of MAD-122 against the hybrid+paged path showed **zero
semantic activity** under a workload that should have produced many
fingerprints. The trigger never fired.

## Decision

The server fingerprints every complete (block-aligned) chunk of a new
seq's prompt at **prefill submission**, before block-table assignment.

Specifically: after the server's `update_slots` builds the prefill
batch and the block table is populated, the server walks the new
seq's complete logical blocks and:

1. Decodes the original tokens for that block via
   `slot.prompt.tokens[]` + `common_token_to_piece`.
2. Calls `mt::llama_memory_tiered::embed_text(text)` for an
   L2-normalized embedding.
3. Calls
   `llama_kv_cache_paged::record_paged_block_fingerprint(seq, lblock, embedding, tier=Hot)`.

Skipped when the block already has a fingerprint (the
`has_paged_fingerprint(seq, lblock)` short-circuit handles re-prefill
of a prior turn's accumulated context).

CPU cost is ~5ms × n_complete_blocks per prefill, off the GPU
critical path. Fingerprint lifecycle is bound to block lifecycle —
they get dropped on whole-seq wipe; per-block removal happens
automatically when a block's table entry is cleared.

## Consequences

**Positive**:
- Trigger fires deterministically on every prefill, observable in
  `paged_semantic_attempts_total`.
- Lives at the server layer where the original token text is still
  available for decoding. The cache doesn't need to persist text.
- Decoupled from the cache's internal eviction policy — the cache can
  evict however it wants without losing fingerprint coverage.

**Negative**:
- Adds ~5ms × n_blocks per prefill of CPU work. For a 2048-token
  prompt at block_size=16, that's ~640ms total. Off the GPU critical
  path but visible in time-to-first-token.
- Sub-block-aligned prompts (< block_size new tokens) don't get
  fingerprinted at all. Only matters for very short prompts.

**Neutral**:
- The cache exposes `record_paged_block_fingerprint` and
  `has_paged_fingerprint` as part of its public API. The wrapper /
  embed path stays in `mt::llama_memory_tiered`
  (see [A8](A8-bge-small-on-wrapper.md)).
