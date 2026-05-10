# A9. Speculative decoding requires real partial seq_rm

**Status**: Accepted (2026-05-10)
**Decided in**: [MAD-126](https://mad-lab-ai.atlassian.net/browse/MAD-126)
**Implemented in**: MAD-128 (paged-compat: real seq_rm middle-range)

## Context

`llama_memory_i::seq_rm(seq, p0, p1)` removes positions `[p0, p1)`
from the seq's KV cache. The interface allows arbitrary middle-range
wipes; speculative decoding relies on this to roll back rejected
draft tokens (the verifier rejects token N, so positions `[N, end)`
get wiped to allow re-decode from a different sample).

The original `llama_kv_cache_paged::seq_rm` implementation treated
**all** non-tail wipes as if they were tail-truncates (wiping
`[p0, end)` regardless of `p1`). This was a silent degradation: the
cache reported success but actually removed more data than asked.
With speculative decoding, that meant losing positions that the
verifier had **accepted** alongside the rejected ones — quietly
producing wrong outputs.

Three choices:

1. **Reject middle-range seq_rm** at the cache boundary, return false.
   Forces speculative decoding to disable itself when paged is active.
2. **Implement real partial seq_rm**. Block-aligned wipes free whole
   physical blocks; sub-block partial wipes leave holes in the
   logical sequence (the kernel handles `kInvalidBlockId` by emitting
   `-INFINITY` logits → zero attention weight; equivalent to the
   token not existing).
3. **Convert to whole-seq wipe**. Simpler than (2) but breaks
   conversation continuity — every speculative rollback would
   blow away the entire prefill.

## Decision

Option 2: implement real partial seq_rm. The cache:

- Block-aligned middle wipes free the wholly-covered physical block(s)
  and reset the logical→physical table entry to `kInvalidBlockId`.
- Sub-block partial wipes (the rare case) leave the block's physical
  storage in place but log a warning. The kernel still emits
  `-INFINITY` for cells outside the live range thanks to per-block
  context-len tracking.
- `seq_pos_max` updates to reflect the new logical extent.

The `kInvalidBlockId` sentinel was already in the design from
[A1's](A1-hybrid-paged-primary.md) BlockTable scheme; partial seq_rm
just leverages it.

## Consequences

**Positive**:
- Speculative decoding works against paged caches without
  degradation.
- Same plumbing supports `mt_record_paged_block_fingerprint`'s
  re-fingerprint path (overwriting an existing fingerprint when the
  block's content gets edited — the partial seq_rm wipes the cells,
  the block_table entry stays, the fingerprint gets overwritten on
  next prefill).
- The kernel's `kInvalidBlockId` handling is now load-bearing in
  multiple places, which is fine because it was always part of the
  paged design.

**Negative**:
- Sub-block partial wipes (`p0 % block_size != 0` and `p1 %
  block_size != 0` for the first/last block) currently log a warning
  and leave the partially-wiped block's data in place. The kernel
  copes via context-len, but it's a code-path that was harder to
  test than block-aligned wipes. Keep an eye on it.

**Neutral**:
- The implementation lives entirely in
  `llama_kv_cache_paged::seq_rm`. The wrapper
  (`mt::llama_memory_tiered::seq_rm`) just delegates.
