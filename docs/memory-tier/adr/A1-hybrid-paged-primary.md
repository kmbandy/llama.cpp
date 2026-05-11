# A1. Hybrid+paged is THE primary path

**Status**: Accepted (2026-05-10)
**Decided in**: [MAD-126](https://mad-lab-ai.atlassian.net/browse/MAD-126)
**Implemented in**: MAD-127 (legacy path removal)

## Context

Three KV-cache implementations existed at the start of the army goal:

1. `llama_kv_cache_paged` — vLLM-style block-indexed paged attention
   for hybrid models (attention + recurrent layers). Active.
2. `llama_kv_cache_tiered` — non-paged tiered cache for pure-attention
   models. Largely a parallel codebase to (1).
3. `server_tiered_cache` — server-side wrapper that brokered tier
   movement for (2). Yet another parallel codebase.

Maintaining three implementations of "tiered KV cache" forked the
work: every new tier feature had to be threaded through three
implementations, every bug fix required deciding which paths it
applied to, and the test surface tripled.

The model landscape is shifting decisively toward hybrid architectures
(Qwen3.5/3.6 family, Mamba-attention hybrids, etc.). User direction
2026-05-10 was unambiguous: "hybrid+paged is THE primary path; pure-
attention is least concern. Basically no one is building pure-
attention models anymore."

## Decision

All new tier features land on `llama_kv_cache_paged`. The legacy paths
are removed:

- `llama_kv_cache_tiered` deleted.
- `server_tiered_cache` deleted.
- `mt::llama_memory_tiered` retained but reduced to a thin shim. Its
  surviving responsibilities are documented in
  [A8](A8-bge-small-on-wrapper.md): bge-small embed model ownership
  and recurrent-state backup on hybrid models.

Pure-attention models still work — they route through
`llama_kv_cache_paged` directly without the hybrid wrapper. They just
don't get any special-cased treatment.

## Consequences

**Positive**:
- One codebase for tier management. Bug fixes apply uniformly.
- Test matrix shrinks. New features need one implementation, not
  three.
- The `BlockPool` / `BlockTable` / `BlockSemanticIndex` primitives
  serve every code path that needs paged-style block management.
- Eliminates the "which cache am I actually getting?" confusion for
  operators and developers.

**Negative**:
- Pure-attention models pay the (small) per-call indirection cost of
  the paged dispatch even when they wouldn't strictly need it.
- The thin `mt::llama_memory_tiered` wrapper is a slight wart —
  carrying it as a separate type for two narrow responsibilities
  (bge-small + recurrent backup). A cleaner long-term move is to fold
  those responsibilities into the model layer directly, but that's a
  bigger refactor than the army goal needs.

**Neutral**:
- Anyone holding stale references to `llama_kv_cache_tiered` or
  `server_tiered_cache` will hit a clean compile error rather than
  silently picking the wrong path.
