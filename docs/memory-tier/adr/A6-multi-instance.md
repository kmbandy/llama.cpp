# A6. Multi-instance isolation = per-instance cold subdir + lockfile

**Status**: Accepted (2026-05-10)
**Decided in**: [MAD-126](https://mad-lab-ai.atlassian.net/browse/MAD-126)
**Implemented in**: MAD-131 (per-instance cold subdir + lockfile + SSD quota + per-machine boot scripts)

## Context

The army goal is four `llama-server` instances per box (one per GPU)
in the steady state. They share a host filesystem and may share a
single SSD configured via `--kv-tier-ssd-path`. Without isolation,
they would:

1. Write to the same per-layer K/V files (`L0.k.bin`, `L0.v.bin`,
   etc.). Different cache geometries between instances would corrupt
   each other.
2. Race on the cold-tier sidecar.
3. Re-open each other's files on cold-resume, treating another
   instance's persisted state as their own.

Three isolation models were possible:

1. **Disjoint paths**: each instance gets its own
   `--kv-tier-ssd-path`. Operationally tedious — operators have to
   provision N directories and remember which is which.

2. **Shared path + namespacing**: one root path, every instance
   writes under a per-instance subdirectory. Operators provision one
   path; the server handles namespacing.

3. **Single shared cache across instances**: one cache backing all
   instances. Architecturally cleanest in some sense but requires
   cross-process coordination (shared memory, IPC) that the
   single-threading contract ([A4](A4-single-threading.md)) rules
   out.

## Decision

Option 2: shared `--kv-tier-ssd-path` with per-instance namespacing.

Each instance writes cold-tier files under
`${ssd_path}/paged/instance-${INSTANCE_ID}/`:

```
${ssd_path}/paged/
├── instance-main-r9700/
│   ├── L0.k.bin
│   ├── L0.v.bin
│   ├── ...
│   ├── instance.lock        # flock'd while the process is alive
│   └── index.bin            # CIDX v1 sidecar
├── instance-main-6900xt/
│   └── ...
```

`--instance-id` defaults to the process PID but is typically set
explicitly by boot scripts so it's stable across restarts (which
matters for cold-resume — see [A5](A5-persistence-explicit.md)).

A `flock`-based lockfile on `instance.lock` prevents accidental
double-start: a second process trying to open the same instance subdir
fails fast with a clear error rather than silently corrupting on-disk
state.

Optional `--kv-tier-cold-budget-mb` caps each instance's cold pool
size in MiB (K+V across all attn layers) so a runaway instance can't
fill the shared disk and starve siblings.

## Consequences

**Positive**:
- One operator-provisioned path serves the whole army on a box.
- Per-instance isolation prevents cross-instance corruption by
  construction.
- The lockfile gives clean error messages on configuration mistakes
  (e.g. operator forgot to change `--instance-id` after copy-pasting
  a boot command).
- Per-instance cold-resume works without any cross-process
  coordination — each instance reads its own subdir.

**Negative**:
- An operator who really wants single-shared-cache semantics (some
  exotic deployment) doesn't get it. Acceptable because nobody's
  asking.
- A stale lockfile from a hard crash blocks restart until cleaned.
  The runbook includes the recovery procedure (verify with `lsof`
  that no process holds it; then `rm`).

**Neutral**:
- Disk-space accounting per instance becomes the operator's
  responsibility. The server reports its own usage in metrics; total
  disk usage across all instances requires `du` or equivalent.
