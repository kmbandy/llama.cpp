# Architecture Decision Records — tiered KV cache

Each file documents one of the resolved design decisions from
Epic MAD-126. Format follows the standard ADR template (Context,
Decision, Consequences). The decisions are referenced by their
A-number throughout [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

| # | Title |
|---|---|
| [A1](A1-hybrid-paged-primary.md) | Hybrid+paged is THE primary path |
| [A2](A2-fingerprint-at-prefill.md) | Semantic fingerprint write trigger = prefill time, not eviction time |
| [A3](A3-semantic-prefetch-only.md) | Semantic drives prefetch only, not eviction |
| [A4](A4-single-threading.md) | Single-threading contract is explicit |
| [A5](A5-persistence-explicit.md) | Persistence is explicit save/restore, no implicit crash recovery |
| [A6](A6-multi-instance.md) | Multi-instance isolation = per-instance cold subdir + lockfile |
| [A7](A7-paged-default-on.md) | Auto-default `--kv-tier-paged-blocks` for hybrid models |
| [A8](A8-bge-small-on-wrapper.md) | bge-small ownership stays on `mt::llama_memory_tiered` |
| [A9](A9-real-partial-seq-rm.md) | Speculative decoding requires real partial seq_rm |
| [A10](A10-real-seq-cp-cow.md) | seq_cp CoW required for branching agent workflows |
