# Army boot scripts (MAD-131)

Templates for spinning up the agent army across the per-machine targets:

- `main.sh` — main box (R9700 + 6900XT). Two instances.
- `mad-lab.sh` — mad-lab box (1070 + RX 480 in ROCm 6.3 docker). Two instances.
- `cleanup-cold.sh` — walks the cold-tier root and removes orphaned `instance-*` subdirs.
- `army.service.example` — systemd unit template.

These are templates. **Customize before use** — paths, model files, ports, and
GPU device IDs all need to match your local setup.

## What "instance" means here

Each `llama-server` process gets a unique `--instance-id` so multiple
processes can share one `--kv-tier-ssd-path` without colliding on
cold-tier files. The cache writes to:

```
${ssd_path}/paged/instance-${INSTANCE_ID}/
  L0.k.bin
  L0.v.bin
  ...
  .lock      (flock for double-start refusal)
  .pid       (informational; current pid)
  index.bin  (cold-index sidecar, written on /slots/save)
```

If two processes try to start with the same `--instance-id`, the second
will refuse with a clear error message naming the holder pid.

## Operator runbook (short version)

- Stop everything cleanly: `systemctl stop army@<instance>`. The cache
  destructor releases the lock, so the next boot can reuse the same ID.
- After an unclean shutdown / crash: the lock is auto-released when the
  process exits (kernel-level flock). The `.pid` file may be stale;
  it's overwritten on next start.
- To wipe cold tier and start fresh: `./cleanup-cold.sh /path/to/ssd`.
