# A10. seq_cp CoW required for branching agent workflows

**Status**: Accepted (2026-05-10)
**Decided in**: [MAD-126](https://mad-lab-ai.atlassian.net/browse/MAD-126)
**Implemented in**: MAD-128 (paged-compat: real seq_cp with refcount-based CoW)

## Context

`llama_memory_i::seq_cp(src, dst, p0, p1)` copies positions
`[p0, p1)` from `src` to `dst`. Branching agent workflows depend on it
heavily: a planner clones the current conversation N times to explore
N hypothetical continuations, then either commits the best branch or
discards them all.

The original `llama_kv_cache_paged::seq_cp` was a no-op that logged a
warning ("seq_cp not yet supported on paged cache"). Branching
workflows just didn't work — `dst` ended up empty and the model
re-prefilled from scratch on every branch, losing all the
conversation context.

A literal byte-copy implementation would have worked but been wildly
expensive: an N-way branch with K-token shared context costs `N × K
× per_token_K_plus_V` extra bytes copied per fork. For a 128k-token
context with 4 branches, that's hundreds of MB of host↔device
copies per fork.

## Decision

Implement copy-on-write via the existing `BlockPool` refcount
machinery:

- `seq_cp(src, dst, p0, p1)` walks `src`'s logical blocks covering
  `[p0, p1)`. For each, the cache:
  1. Calls `pool_.bump_ref(physical_block_id)` — refcount++.
  2. Calls `table_.append_block(dst, physical_block_id)` — `dst`'s
     logical sequence now points at the same physical block.
- `pool_.free_block` decrements refcount; only returns the block to
  the free stack when refcount drops to 0. So a shared block stays
  alive as long as ANY seq references it.

When `dst` later writes new tokens to a shared block (the cell
positions diverge from `src`'s view), the cache detects shared
status via `pool_.refcount(physical) > 1` and allocates a fresh
block, memcpys the existing data into it, swaps the table entry, and
decrements the old block's refcount. That's the copy-on-write step.

## Consequences

**Positive**:
- Branching is essentially free at fork time — one block-table
  copy + N refcount bumps. The N×K bytes get copied only when (and
  if) a branch actually mutates that block.
- Multi-path planning workflows become viable on paged caches.
- The refcount machinery was already in `BlockPool` for other
  reasons (the eviction path needed to know which blocks were live
  vs. truly free); adding `bump_ref` and refcount-aware
  `free_block` was minimal incremental work.

**Negative**:
- Eviction has to skip refcount > 1 blocks during victim selection,
  otherwise it'd evict a block another seq depends on. Current
  behavior: if the LRU candidate is shared, skip and try the next.
  Could thrash if every hot block is shared — pathological but
  recoverable (operator sees `paged_evict_hot_to_warm_total` rate
  spike).

**Neutral**:
- The CoW write trigger lives inside `llama_kv_cache_paged`'s prefill
  path. Anyone changing how new blocks get written needs to preserve
  the `if (refcount > 1) clone-then-write` check.
