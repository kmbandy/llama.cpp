# A5. Persistence is explicit save/restore, no implicit crash recovery

**Status**: Accepted (2026-05-10)
**Decided in**: [MAD-126](https://mad-lab-ai.atlassian.net/browse/MAD-126)
**Implemented in**: MAD-130 (state_write/state_read on paged + cold resume + fingerprint save/load)

## Context

The cache holds substantial state — block table, hot/warm/cold tier
contents, semantic fingerprints. Two persistence models were
candidates:

1. **Implicit crash recovery**: write enough state continuously to
   disk that a sudden process death leaves the cache in a recoverable
   state on restart. Requires WAL-style logging or transactional
   block writes.

2. **Explicit save/restore**: serialize state on demand via the
   server's `/slots/save` endpoint; restore via `/slots/restore` on
   the next boot. A clean shutdown saves; a crash leaves nothing
   recoverable.

The tradeoffs run opposite directions on every axis:

| Axis | Implicit | Explicit |
|---|---|---|
| Operator complexity | Low (just works) | Medium (call /slots/save) |
| Implementation complexity | High (WAL semantics, partial-write recovery, format versioning, double-write throughput cost) | Low (one serializer, one deserializer) |
| Crash semantics | "Best effort recovery — please report bugs" | "Crash = blank slate; clean shutdown = restored" |
| Performance | Continuous disk pressure | Zero disk pressure between saves |

## Decision

Explicit save/restore via the server's existing `/slots/save` and
`/slots/restore` endpoints. Crash recovery is **not** automatic.

State written by `state_write()`:
- Block table per seq (logical→physical mapping; PAGS v1 magic).
- Hot-tier K/V tensors (or skip + reprefill — caller-configurable
  via the `flags` argument).
- Warm-tier K/V buffers.
- Cold-tier index sidecar (CIDX v1 magic = `0x58444943`).
- BlockSemanticIndex fingerprints (PSFI v1 magic = `0x49465350`).

Format magic numbers and version bytes let future format changes be
detected and rejected (rather than silently corrupting state).

The cold-tier sidecar is the recovery hinge: per-layer K/V files
contain the actual block data, but without the sidecar's `(seq,
lblock) → file_offset` mapping the cache can't read them back. Hard
crashes that miss the sidecar write produce orphan files which
`scripts/army/cleanup-cold.sh` removes.

## Consequences

**Positive**:
- Implementation is straightforward. One serializer per state
  category; one set of unit + integration tests.
- Zero steady-state disk overhead between saves. Operators control
  when (and whether) to save.
- Format versioning gives a clean upgrade story. Adding a new field
  bumps the version byte; old loaders refuse cleanly.

**Negative**:
- Crash recovery is "start over." For long-running servers with
  large cold tiers, that's a meaningful cost — the cold pool gets
  rebuilt only as the workload re-prefills the same content.
- Operators must remember to call `/slots/save` (or wire it into
  their service's `ExecStop`). A box that gets rebooted without that
  call loses everything cold.

**Neutral**:
- The cold-tier per-layer files survive on disk after a crash, but
  without the sidecar they're unreadable. They're not corrupting
  anything; they're just dead weight until cleaned up.
- A future "crash-recoverable" mode is not precluded — it would be a
  separate feature on top of this one (probably a write-ahead log of
  block-table mutations).
