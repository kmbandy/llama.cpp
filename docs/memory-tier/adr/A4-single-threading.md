# A4. Single-threading contract is explicit

**Status**: Accepted (2026-05-10)
**Decided in**: [MAD-126](https://mad-lab-ai.atlassian.net/browse/MAD-126)
**Implemented in**: MAD-132 (concurrency hardening)

## Context

`llama_kv_cache_paged` and its `BlockPool` / `BlockTable` /
`BlockSemanticIndex` companions hold mutable state. None of them have
internal synchronization (mutexes, atomics on counters, etc.). The
question was whether to add internal locking, document a single-
threading contract, or design for explicit concurrency.

vLLM, the architectural reference, runs its scheduler single-
threaded for exactly the same reasons we're considering single-
threading: cache state machines are easier to reason about
sequentially, locking adds latency to the hot path, and the workload
(one batch per scheduler tick) doesn't naturally benefit from
parallelism inside the cache itself.

## Decision

Single-threading is the **explicit contract**. Document it in the
class headers; add debug-build assertions to catch violations.

Concretely:

- The server's main `update_slots` loop is the single mutator. HTTP
  worker threads communicate with it via `server_queue`'s task
  channel and never touch the cache directly.
- Tier-counter `_total` accessors return `uint64_t` from the cache.
  These are read by the `/metrics` HTTP handler from a different
  thread — this is the **only** sanctioned cross-thread cache
  access. It's safe-ish on x86_64 and aarch64 for monotonic 64-bit
  counters; aligned-uint64 atomic loads are single instructions
  on those ISAs, so torn reads aren't a concern.
- Debug builds include `check_thread_id_()` in
  `llama_kv_cache_paged` that traps if anyone other than the
  registered main thread mutates the cache.
- Any future async work (semantic prefetch on a worker, batched
  bge-small embed off the critical path) is gated on a real
  concurrency design — not "by accident."

## Consequences

**Positive**:
- Reasoning about cache state is straightforward — no interleavings,
  no data races, no lost-update bugs.
- No locking overhead in the hot path. Eviction, allocation, and
  fingerprint scoring are all single-instruction-stream operations.
- Test surface stays manageable; we don't need to write race tests
  for code paths that can't race by contract.

**Negative**:
- Constrains future work. Any feature that wants to run on a worker
  thread must either move the work outside the cache (compute on the
  worker, hand the result back to the main thread to apply), or
  carry a real concurrency-design proposal to add minimal locking.
- The cross-thread metrics read is technically a contract violation
  — it works in practice but it's load-bearing on aligned-64-bit
  atomicity. If the codebase ever ports to a 32-bit platform, the
  counters need to be split or made atomic.

**Neutral**:
- The contract is unenforceable at runtime in release builds. Anyone
  introducing a multi-threaded mutation path will have to cross-check
  the assertion in debug builds — relying on developer discipline to
  do so.
