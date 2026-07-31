# Design: host-RAM victim tier for the expert worker

Status: design, ready for implementation
Author: Claude (design/review). Implementation to gpt-5.6-terra.
Date: 2026-07-30

## 1. The gap

`tools/wp-expert-worker/wp-expert-worker.cpp` has **no host-side page retention at all**.
`select_victim()` (line 1083) chooses a *VRAM slot* to overwrite; the evicted page is
simply gone. Every subsequent miss on that page re-reads ~12.2 MB from NVMe.

Verified: `grep -cE "host_pages_|host_cache|spill|demote"` returns **0**.

Meanwhile, measured tonight:

| worker | miss rate | read ms/expert | effective GB/s | drive rating |
|---|---|---|---|---|
| R9700 `:8801` | 46.0% | 1.012 | 5.55 | 6.25 (89%) |
| 1070 `:8803` | 53.4% | 2.965 | 2.20 | shared SN550 ~3.08 |
| RX 480 `:8804` | 49.4% | 3.504 | 1.72 | shared SN550 ~3.08 |

**mad-lab-2026's combined read is 3.92 GB/s against a ~3.08 GB/s drive — saturated.
mad-lab-main is at 89% of rating.** Both machines are storage-constrained while
serving ~50% miss rates.

And both have **~9–10 GB of system RAM sitting entirely unused**.

## 2. What to build

Wire the existing `HostTier` (`src/weight-pager/wp-host-tier.h`, *"optional
pinned/pageable host RAM cache for weight pages"*) into the expert worker as a victim
tier behind the VRAM slots.

**Reuse it. Do not write a new cache.** This mechanism is what produced a previous
single-machine throughput doubling (1.736 → 3.570 tok/s, perplexity unchanged).

Flow:

```
request -> VRAM slot hit?          -> serve (unchanged)
        -> HOST TIER hit?          -> H2D upload, promote to VRAM slot   [NEW]
        -> miss                    -> NVMe read -> H2D -> VRAM slot
eviction from VRAM  -> DEMOTE the page into the host tier  [NEW]
                       (instead of dropping it)
```

### 2.1 Budget and flags

- New flag `--host-victim-bytes N`, plus env `WP_EXPERT_HOST_VICTIM_BYTES`.
- **Do NOT overload `--host-budget-bytes`.** That already means the *staging arena*
  (the 16-buffer O_DIRECT pool, ~249 MiB, reported as `host_budget=` in the worker's
  listen line). Conflating the two would silently resize staging. They are separate
  budgets with separate purposes.
- Default **off** (0 bytes), so existing behaviour is unchanged unless requested.

### 2.2 Host RAM arithmetic — mandatory, this box has been OOM-killed before

mad-lab-2026 has **15 GB total, ~10 GB available**, and runs two workers *plus* live
fleet services. Proposed starting point:

```
2 workers x 2.5 GB victim tier   = 5.0 GB
staging arenas 249 MiB x 2       = 0.5 GB
                                   -------
                                   5.5 GB on top of ~5 GB already used = ~10.5 / 15 GB
```

Leaves ~4 GB headroom. **Do not exceed this without re-doing the arithmetic.** Note the
known cliff: above roughly 7/8 of machine RAM, OS paging turns cache "hits" into page
faults and throughput collapses (an external engine measured 0.32 → 0.04 tok/s falling
off this edge). Staying well under is the point.

## 3. THE LANDMINE — read this before touching HostTier

`src/weight-pager/` has produced the **same class of bug six times**: code guarded by
`#if defined(GGML_USE_CUDA)` / `GGML_USE_HIP` that is compile-time satisfied but
runtime-wrong for the active backend.

The specific precedent here is exact: the RAM victim tier once contained a raw
`hipMemcpy` D2H under a `GGML_USE_CUDA` guard that *was* satisfied on Vulkan-only
builds — so enabling the host budget would have copied from the **Vulkan sentinel
pointer** on every eviction. It was fixed (6dedabb5f), but the pattern recurs.

**The expert worker runs on THREE backends: ROCm (R9700), CUDA (GTX 1070), and Vulkan
(RX 480).** Any device→host or host→device copy in this path must dispatch on the
*runtime* backend, never on a compile-time macro. Route through the existing
backend-neutral transport (`ggml_backend_tensor_get/set` or the pager's
`GpuTransport`) rather than calling `hipMemcpy`/`cudaMemcpy` directly.

If you find yourself writing `#if defined(GGML_USE_...)` in this change, stop and
report it.

## 4. Required counters — a mechanism counter, not just an outcome

Add to the `WP_WORKER_STATS=1` line:

```
n_host_hit      pages served from the host tier
n_host_demote   pages written into the host tier on eviction
ns_host_get     time spent serving host-tier hits
host_bytes      bytes resident in the tier
```

**`n_host_hit` is the mechanism counter and it is not optional.** A throughput change
with `n_host_hit = 0` means the tier never engaged and any improvement came from
somewhere else. We have shipped a "+3.7%" result before that turned out to be a
cold-cache artifact with the mechanism counter reading exactly zero — the counter is
the only reason that was caught.

## 5. Expected effect, written down in advance

Stated now so a mismatch is a finding rather than something to rationalize:

- A host-tier hit still costs an H2D upload (~12.2 MB over PCIe ≈ 1 ms), so it is **not
  free** — it saves the NVMe read, not the transfer.
- On 2026 that trades a ~3.0–3.5 ms read for a ~1 ms upload: **~2–2.5 ms saved per
  converted miss**.
- 2.5 GB ≈ 205 additional pages per worker, against ~500 VRAM slots — roughly **+41%
  effective capacity**.
- Hit-rate response to capacity is **not linear** and is deliberately not predicted here.
- If `n_host_hit` is large but throughput is flat, the H2D leg is eating the gain and the
  next question is upload cost, not cache size.

## 6. Verification

1. **Mechanism**: `n_host_hit > 0` and `n_host_demote > 0`. If either is zero, stop.
2. **Miss rate**: NVMe miss rate should fall from ~50%. Report before/after.
3. **Control**: tier off vs tier on, same workload. Run-to-run variance on this workload
   is **±3%**, so a change below ~5% is not a result — repeat runs before claiming one.
4. **Correctness**: output must stay coherent. Do **not** use output sha256 as the check —
   greedy argmax masks small numeric differences, and output was bit-identical across
   moving 6 GiB of weights between devices today.
5. **Host RAM**: watch actual RSS against the §2.2 budget on both machines.

## 7. Constraints

- Build **both** `build-hip` (main) and `build-army` (2026), libllama and llama-server
  together, `-j2` on 2026 (15 GB RAM, i7-6700K).
- mad-lab-2026 runs live fleet services from `build-army` (pid 855466 nemotron embedder,
  pid 3025042 llama-router). Move the active `libllama.so*` chain aside before rebuilding
  so running processes keep their inode; do not signal or restart them.
- No inference or GPU work by the implementer — Claude runs all measurements.
