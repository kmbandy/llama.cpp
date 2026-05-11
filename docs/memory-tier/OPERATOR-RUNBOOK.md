# Tiered KV cache — operator runbook

Day-2 operations for the army stack: four `llama-server` instances
across four GPUs, each running paged + tiered with semantic prefetch.
This document is the playbook for "what do I do when X happens at 2am."

For end-user docs see [USER-GUIDE.md](USER-GUIDE.md). For internals
see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Topology

| Instance ID | Box | GPU | Build | Model |
|---|---|---|---|---|
| `main-r9700` | main dev box | R9700 (gfx1201, 32 GB) | `build-hip` | Qwen3.6-27B-Q6_K |
| `main-6900xt` | main dev box | 6900XT (gfx1030, 16 GB) | `build-hip` | Qwen3.5-9B-TQ3_1S |
| `mad-lab-1070` | mad-lab-2026 | GTX 1070 (sm_61, 8 GB) | `build-army` (CUDA) | omnicoder-9b-q5_k_m |
| `mad-lab-rx480` | mad-lab-2026 | RX 480 (gfx803, 8 GB) | `build-rocm-gfx803` (in `rx480-army` docker) | omnicoder-9b-q5_k_m |

Per-machine boot scripts live under `scripts/army/`:

- `scripts/army/main.sh` — main dev box (r9700 + 6900xt)
- `scripts/army/mad-lab.sh` — mad-lab box (1070 + rx480)
- `scripts/army/cleanup-cold.sh` — wipes the cold-tier directory
- `scripts/army/army.service.example` — systemd template

---

## Boot

### Cold start

```bash
# Main box
bash scripts/army/main.sh

# mad-lab
ssh mad-lab-2026 'bash ~/GitHub/llama.cpp/scripts/army/mad-lab.sh'
```

Each script launches its instances in the background, writes pids and
log paths to stdout, and returns. Boot is considered complete when all
instances log `main: server is listening on http://…`.

### Resume after a clean shutdown

If the prior shutdown went through `/slots/save` (or systemd's
`ExecStop`), add `--kv-tier-cold-resume` to each instance's invocation.
The cold-tier files are re-opened and the cold index sidecar is
replayed. Without it the cold pool starts empty.

### Hard restart (after crash)

The cold-tier sidecar (`paged/instance-${ID}/index.bin`) is only
written on a graceful shutdown. After a crash the cold pool starts
empty regardless of `--kv-tier-cold-resume`; the per-layer files are
still on disk but with no live index they're inaccessible. If you want
clean disk hygiene, run `bash scripts/army/cleanup-cold.sh` before
restart to delete the orphan files.

---

## Health checks

### Per-instance

```bash
# Aliveness
curl -sf http://127.0.0.1:11435/health | jq

# Tier counters (Prometheus format; needs --metrics)
curl -s http://127.0.0.1:11435/metrics | grep -E 'paged_(evict|restore|semantic|seq|blocks)'

# Per-seq tier residency
curl -s http://127.0.0.1:11435/slots | jq '.[] | {id, state, paged_tier}'
```

### Fleet-wide

```bash
for port in 11435 11436 11437 11438; do
    echo "=== :$port ==="
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && echo OK || echo DOWN
done
```

---

## Common alarms

### Cold pool occupancy > 80%

**Symptom**: `paged_blocks_capacity_cold` is fixed; the count of
unmapped (free) cold blocks is dropping toward zero. Visible by
diffing two `/metrics` snapshots, or by watching
`paged_evict_cold_to_drop_total` start to increment.

**Action**:
1. Confirm the workload — is something pinning a lot of context that
   isn't being released? Check `/slots` for tasks that have been
   `is_processing=true` for an unusually long time.
2. If load is legitimate, raise `--kv-tier-cold-budget-mb` and restart
   the affected instance with `--kv-tier-cold-resume` to preserve what
   you have.
3. If load is a stuck task, kill the slot via the server's task API or
   restart the instance.

### `paged_evict_cold_to_drop_total` is non-zero

**Symptom**: cold pool is full enough that the cache started **dropping
data** (not just spilling). Future queries against those positions get
`-INFINITY` logits — the kernel will try and silently produce nonsense.

**Action**: this is a sizing failure. Either widen `COLD%`, raise
`--kv-tier-cold-budget-mb`, or reduce `-c` so the workload fits the
configured tiers. **Do not ignore this counter.**

### Semantic hit rate < 10%

**Symptom**: `paged_semantic_hits_total / paged_semantic_attempts_total`
< 0.10 across many requests.

**Possible causes**:
- The workload doesn't revisit content (every turn is fresh prompt).
  Semantic prefetch isn't useful here; consider disabling
  `--kv-tier-semantic-index` to save the per-block embed cost.
- `--kv-tier-semantic-threshold` is too strict. Try lowering from 0.65
  to 0.55.
- The bge-small model failed to load. Check the boot log for
  `mt::EmbeddingModel: failed to load model`. Common fix: re-download
  the gguf or use a different embedding model.

### Eviction rate > 100/sec sustained

**Symptom**: `paged_evict_hot_to_warm_total` rate is climbing fast.

**Action**: hot pool is undersized for the workload, or the hot
percentage is too low. Either raise `HOT%` in `--kv-tiered` or reduce
`--parallel` so each agent's working set has more room. Per-batch
eviction events are normal; sustained high rate hurts decode latency.

### MAD-141: "paged KV admission could not fit a N-token request"

**Symptom**: client receives HTTP 500 with this body. Server log shows
`MAD-141: paged admission stuck for 4 iters with no eviction progress`.

**Action**: the prompt is genuinely too large to fit alongside the
live workload. Options:
- Caller sends a smaller prompt (chunk + use `cache_prompt: true`).
- Wait for sibling slots to release their blocks (they finish decoding,
  client closes the connection).
- Operator: raise `-c` so the server has more total blocks, or reduce
  `--parallel` so each slot has a larger share.

This is a **graceful** failure mode (post-MAD-141 / commit `0d66d8aa3`).
Pre-fix builds would crash the server entirely; if you see a process
abort with `n_empty_consecutive > 3`, you're on a stale build.

### Lockfile error on startup

**Symptom**: `instance lock /…/instance.lock held by another process`.

**Action**: another `llama-server` is already running with the same
`--instance-id` against the same `--kv-tier-ssd-path`. Either:
- Kill the previous instance (`pgrep -af "instance-id <ID>"` →
  `kill <pid>`).
- Pick a different `--instance-id` for this one.
- If the previous instance crashed and left a stale lockfile,
  `rm /…/instance.lock` then restart. Verify with `lsof` first that
  no process actually holds it.

---

## Manual interventions

### Clean restart of a single instance

```bash
INSTANCE=main-r9700
PORT=11435

# Graceful shutdown via SIGTERM. Server will flush cold sidecar.
pkill -TERM -f "instance-id $INSTANCE"

# Wait for socket to release.
while ss -ltn | grep -q ":$PORT"; do sleep 1; done

# Restart with resume.
bash scripts/army/main.sh   # or just the relevant chunk of it
```

### Cold-tier wipe (drop all SSD state)

```bash
INSTANCE=main-r9700
SSD_PATH=/var/lib/army/ssd

# Stop the instance first.
pkill -TERM -f "instance-id $INSTANCE"

# Delete this instance's cold-tier dir.
rm -rf "$SSD_PATH/paged/instance-$INSTANCE"

# Restart without --kv-tier-cold-resume.
bash scripts/army/main.sh
```

For a fleet-wide cold wipe use `scripts/army/cleanup-cold.sh`. Stop
all instances first.

### Fingerprint reset (drop semantic prefetch state)

Fingerprints are stored alongside the live cache state and are dropped
on whole-seq `seq_rm` automatically. To force-drop without losing the
KV state, the cleanest path is:

1. Save state via `/slots/save`.
2. Restart the instance (without `--kv-tier-semantic-index` if you
   want to permanently disable semantic prefetch, or with it for a
   fresh fingerprint generation).

Per-block fingerprint deletion isn't exposed via HTTP; it requires a
custom server build that calls `paged_semantic_.clear()`.

---

## Disaster recovery

### Lost SSD (cold-tier dir disappeared)

The hot and warm tiers are fine — they're in VRAM and host RAM. Cold
data is gone forever. The server will keep serving live requests; only
queries against positions that had been spilled to cold will read
sentinel logits (manifesting as garbage tokens for those positions).

Action:
1. Recreate the directory and ensure permissions are correct.
2. Restart the instance **without** `--kv-tier-cold-resume`. Cold pool
   starts fresh.
3. If client sessions need their cold-evicted context back, they'll
   have to re-prefill it (i.e. resend the original long prompt).

### Disk full on `--kv-tier-ssd-path`

Cold writes start failing silently after the first ENOSPC. The cache
can't tell the difference between "wrote successfully" and "filesystem
silently dropped this." Eventually the server hits the cold pool's
configured size, can't drop further, and starts logging
`paged_evict_cold_to_drop_total` ticks.

Action:
1. Free space on the disk (rotate logs, drop other temp data).
2. If urgent, reduce `--kv-tier-cold-budget-mb` and restart so the
   server tries to use less.
3. Long-term: move `--kv-tier-ssd-path` to a larger disk.

### GPU OOM on startup

Server fails with `cudaMalloc failed: out of memory` or
`alloc_tensor_range: failed to allocate ROCm0 buffer`.

Diagnose:
```bash
# AMD
rocm-smi --showmeminfo vram --showpids

# NVIDIA
nvidia-smi
```

Common causes:
1. **Stale process holding VRAM**. Kill it (`kill -9 <pid>`). Common
   on AMD with ungraceful shutdowns where the kernel module doesn't
   reclaim immediately.
2. **HOT% sized too high** for the model's weight footprint. Reduce
   `HOT%` in `--kv-tiered`, raise the warm/cold percentages.
3. **Other instance on the same GPU**. Confirm via `--device` mapping
   that two `llama-server` instances aren't pointing at the same card.

### Whole army down after kernel/driver update

Symptoms: every instance fails to start with HIP/CUDA initialization
errors.

Action:
1. Check `dmesg` for amdgpu/nvidia driver load failures.
2. Verify `rocm-smi` (AMD) and `nvidia-smi` (NVIDIA) both work.
3. Verify the binary's compiled-for arches still match the GPUs:
   ```
   strings llama-server | grep -E 'gfx[0-9]+|sm_[0-9]+'
   ```
   If a driver update bumped the GPU's minimum-supported arch beyond
   what the binary targets, rebuild with the new arch in
   `-DAMDGPU_TARGETS` / `-DCMAKE_CUDA_ARCHITECTURES`.

---

## Per-device caveats (from MAD-136 verification)

### R9700 (gfx1201, RDNA4)

- Build with `-DAMDGPU_TARGETS=gfx1201` (or include it in a multi-arch
  build).
- Stable on ROCm 6.4+; earlier ROCm versions don't recognize gfx1201.
- Decode rate baseline: ~22 t/s on Qwen3.6-27B-Q6_K with
  `--parallel 1 -c 8192` and turbo4 KV.

### 6900XT (gfx1030, RDNA2)

- Build must include `-DAMDGPU_TARGETS=gfx1030`. **Silent crash** if
  missing — HIP runtime emits "No compatible code objects found for
  gfx1030" before llama.cpp's logger runs. Diagnose with
  `AMD_LOG_LEVEL=3`.
- Runs as eGPU over Thunderbolt 3 in this fleet; PCIe bandwidth limits
  apply but don't significantly affect inference (model load is the
  one-time cost).

### GTX 1070 (sm_61, Pascal)

- Build with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61`.
- Compute capability 6.1 — no FP16 tensor cores, no bf16. Stick to
  `f16` or `turbo4` cache types; `bf16` not supported by Pascal.

### RX 480 (gfx803, Polaris)

- **Vulkan path is ~10× slower than ROCm** on Polaris (1.9 vs 19.6
  t/s on omnicoder-9b in MAD-136 verification). Stick with ROCm in a
  docker container (`rocm/dev-ubuntu-24.04:6.4-complete` base, env
  `HSA_OVERRIDE_GFX_VERSION=8.0.3 PYTORCH_ROCM_ARCH=gfx803
  ROC_ENABLE_PRE_VEGA=1`). The `rx480-army` container on
  `mad-lab-2026` is the working setup.
- Cannot run modern AMD-supported kernels natively; stays on the
  custom container indefinitely (or until the box migrates to CachyOS).

---

## Routine maintenance

### Weekly

- `df -h "$SSD_PATH"` — confirm cold-tier disk usage isn't drifting up.
- `journalctl --user -u army-* --since '7d ago' | grep ERROR` — scan
  for warnings the alarms didn't catch.

### Monthly

- Rotate `/var/log/army/*.log` — these grow with verbosity 1 logs.
- Re-run `bash scripts/test/run-army-matrix.sh` against the live
  branch to confirm regressions haven't snuck in.

### After any branch merge that touches `src/llama-kv-cache-paged.*`,
`src/memory-tier/**`, `tools/server/server-context.cpp`

- Rebuild affected boxes.
- Run `ctest -R 'test-mt-|test-paged-'` against each build.
- Run a 60-second smoke per device with the new binary before
  promoting to production.
