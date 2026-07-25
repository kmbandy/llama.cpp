# Tiered Dual-GPU Expert Feeding — DS4-Flash paging redesign

**Date:** 2026-07-21
**Status:** design / pre-build. Open validations listed in §7.
**Context:** supersedes the custom `--no-mmap` + p2p-direct VRAM-slot pager as the
primary decode path. Prompted by ik_llama.cpp PRs 1634 (`--defer-experts`) /
2101 (`--prefetch-experts`) and the session's offline measurements.

## 1. The reframe: DS4 decode is starved, not compute-bound

The two GPUs are memory-bandwidth engines that crank when fed:

| card | link | BW | proven decode |
|---|---|---|---|
| R9700 (gfx1201, 32GB) | large-BAR PCIe (~25 GB/s H2D) | **640 GB/s** | bandwidth-capped, "can't cook on decode" = wants more BW |
| RX 6900 XT (gfx1030, 16GB) | **Thunderbolt 3 (~2.5 GB/s)** | **512 GB/s** | 490–735 t/s high-par, decode kernels at parity w/ plain KV |
| **aggregate** | | **~1.15 TB/s** (~800–1000 GB/s realizable @ par1) | |

DS4-Flash decodes at ~2.4 t/s. Measured cause is **starvation**: experts are read
from NVMe at ~0.8 GB/s (12% of the drive, `avg_n` 5.4 of a 16-deep ring — the
system is CONCURRENCY-bound), single-card, with **no RAM cache** — `--no-mmap` +
p2p-direct DMAs NVMe→VRAM around the OS page cache, so every re-used expert is
re-read from disk. We are feeding a 1.15 TB/s engine through a straw.

## 2. Constraints that dictate the design

- **TB3 forbids streaming weights to the 6900XT.** 4.25 MB expert page over
  2.5 GB/s = 1.7 ms — worse than reading it from NVMe. The 6900XT can only be a
  **compute-in-place** device: its experts are RESIDENT and never move. Only
  ~16 KB activations cross TB3 (~6 µs; ~1.3 MB/token over 40 layers = ~0.5 ms,
  negligible vs a ~400 ms token).
- **RAM is small and shared.** 16 GB box, ~8 GB usable for cache (OS + other
  sessions need headroom). Not 15 GB.
- **Decode is memory-bound GEMV.** Throughput ∝ bandwidth fed → summing both
  cards' BW is the top lever; cutting/reordering NVMe reads is the second.

## 3. Measured evidence (offline, this session, from routing_capture.bin)

- VRAM expert pool @ 24 GB already hits **78.5%**; ~690 MB/token to NVMe
  (validates against the measured ~625 MB/token — the model is trustworthy).
- **+8 GB RAM cache → 522 MB/token (−24% NVMe)**, and those reads leave the
  concurrency-bound io_uring ring entirely (RAM hit = ~25 GB/s H2D on the copy
  engine, not a queued NVMe read). Full 63 GB → −66%.
- **Hot-expert coverage** (top-N (layer,expert) by frequency, ~13.4 MB each):
  8 GB → 31%, 10 GB → 35%, 12 GB → 39%; balanced 44% needs ~15 GB. Moderate
  concentration (top 10% of instances = 41%, top 30% = 71%). CAVEAT: measured on
  ONE prompt ("history of Rome") — must re-measure on the code workload (§7).

## 4. Architecture

Three memory tiers, two compute devices.

```
                 NVMe (151 GB, backing store)   <- Tier 2, touched only on RAM miss
                        |  read into RAM+VRAM
                 host RAM cache (~8 GB, LRU)     <- Tier 1, shared victim tier
                    |  H2D ~25 GB/s (fast link)
        +-----------+------------------------------+
        |                                          |
   R9700 VRAM pool (paged experts, LRU)      6900XT VRAM (PINNED hot experts)
   + its dense/attn shards + KV              + its dense/attn shards + KV
   computes cold experts @ 640 GB/s          computes hot experts IN PLACE @ 512 GB/s
        \                                          /
         \------ ~16 KB activations over TB3 ------/   (per MoE layer, negligible)
```

- **Tier 0 / 6900XT:** dense+KV it owns + **top-N hot experts pinned** (~8 GB).
  Computes them in place; never pages. Hot set is static per workload (adaptation
  would require TB3 weight streaming — forbidden); optionally re-warmed once per
  session/task over TB3, amortized, never per-token.
- **Tier 0 / R9700:** LRU pool of the **cold/churning experts**, paged from Tier 1
  over the fast link. Computes them at 640 GB/s.
- **Tier 1 / RAM:** shared LRU expert cache = `HostTier` (`wp-host-tier.cpp`) —
  pinned `hipHostMalloc` arena, `store`/`lookup` by page_idx, LRU eviction,
  `WP_HOST_BUDGET_BYTES` gated. REAL but **only wired into the serial
  `page_in_sync_` fallback** (wp-pager.cpp:3538 lookup / 3611 store). The fast
  decode path — `ensure_batch` under p2p (wp-pager.cpp:2024-2039) — reads
  NVMe→VRAM direct via multi-QD io_uring and **bypasses HostTier entirely**
  (deliberate: comment 2021-2023 "the throughput case that matters for decode is
  P2P"). COUPLING PROBLEM: p2p leaves no RAM copy to cache, so the demand path
  can't populate HostTier without giving up p2p concurrency. RESOLUTION (ik's
  decoupling): the demand path only READS the cache; an async PREFETCH WORKER
  populates it (NVMe→RAM). So HostTier-on-the-fast-path and the prefetch worker
  are ONE unit, not separable phases.
- **Tier 2 / NVMe:** full model; only read on RAM-tier misses.

**Per-token compute:** router picks 6 experts/layer; partition by residency
(hot→6900XT, cold→R9700); both cards compute their share **in parallel**; the
memory-bound expert step runs against ~800–1150 GB/s instead of 640.

## 5. Why this succeeds where speculative prefetch failed

Cross-layer speculative prefetch (this session, refuted) ADDED bytes to a
concurrency-starved ring and lost. This design does the opposite on every axis:
the RAM cache **subtracts** reads from the ring; the dual-GPU split **adds
bandwidth**; the 6900XT hot set **removes** its experts from the paging path
entirely. No speculation, no prediction, no wrong-page race — it's caching +
placement, correct by construction.

## 6. Expected gains (first-order, stacked)

- RAM tier: −24% NVMe reads (8 GB), off the concurrency-bound ring.
- Dual-GPU expert step: ~1.25–1.55× @ par1 (800–1000 GB/s), up to ~1.8× if the
  hot set reaches the 44% balance point.
- End-to-end diluted by attention/router/sampling, but the expert GEMV is the
  dominant memory-bound cost — the thing all the paging exists to feed.

## 7. Open validations (gate the build)

1. **Hot-set generalization on the CODE workload** (GPU: capture routing traces
   on representative code prompts; measure top-N overlap + coverage). The one
   real risk — MoE has a persistent hot core, but we measured the wrong domain.
2. **Single-stream bandwidth saturation** (GPU: confirm par1 expert GEMVs pull
   ~800–1000 GB/s across both cards).
3. **Dual-device dispatch plumbing** — route per-token experts to two devices,
   overlap compute. `-ot` per-tensor placement is the nearest existing hook;
   per-*token* dynamic 2-device dispatch is new. Biggest engineering risk.
4. **`wp-host-tier.cpp` audit — DONE (2026-07-21).** HostTier is real (arena +
   store/lookup + LRU) but wired only to the serial `page_in_sync_` fallback; the
   fast p2p `ensure_batch` path bypasses it (§4 Tier 1). Not a config-enable — it
   needs concurrent-path integration + a prefetch worker to populate it.

## 8. Build phases (rough, each independently measurable)

- **Phase 0 — validate (cheap GPU):** run buffered transport (p2p off) +
  `WP_HOST_BUDGET_BYTES=~8G`, read `host_tier_hits` + NVMe bytes. Confirms the
  sim's −24% hit rate on real hardware. NOT a throughput test (the buffered path
  is serial — comment wp-pager.cpp:2021-2023); it de-risks Phase 1's premise.
  Also §7.1 (hot-set on code prompts), §7.2 (single-stream saturation).
- **Phase 1 — HostTier on the concurrent path (code):** add `host_tier_->lookup`
  to the p2p `ensure_batch` miss path (RAM hit → batched H2D; RAM miss → p2p
  io_uring). Measure decode t/s + NVMe reduction.
- **Phase 2 — async prefetch worker (code):** borrow ik_llama's worker-pool +
  selective-routed-id enqueue (ggml-moe-prefetch.cpp), targeting
  `HostTier.store()` (NVMe→RAM) instead of MADV on mmap. Reactive decode +
  lookahead prefill. Makes the RAM tier active. Misprediction is now free (wrong
  guesses waste RAM only — no VRAM pollution, no race). REQUIRED: HostTier has NO
  internal locking (single-threaded contract), so the worker calling store/lookup
  concurrently with the eval-thread commit-loop store() WILL race its maps — add
  a mutex to HostTier (or a lock-free single-producer handoff) as part of Phase 2.
- **Phase 3 — 6900XT hot-set + parallel dual-device dispatch:** static-pin top-N
  experts on the 6900XT (compute-in-place), overlap both cards' expert compute.
  Measure realized aggregate bandwidth vs the ~800–1000 GB/s (par1) target.

## 8a. Implementation status (2026-07-21)

- **Phase 1 CODE DRAFTED** (uncommitted, in `ensure_batch`'s `WP_ENSURE_BATCH_HOST`
  O_DIRECT block, wp-pager.cpp): Tier-1 lookup partition (RAM hit → H2D from the
  pinned arena, skip the storage read) + populate `HostTier.store()` from fresh
  reads. Single-threaded (no worker yet), flag-gated (default path byte-identical).
  CODE-REVIEWED CLEAN (2026-07-21): 8 correctness items verified; one finding
  applied — `hipDeviceSynchronize()` return was unchecked (silently-wrong-weights
  class); now a sync failure forces the whole batch to the `page_in_sync_`
  fallback. NOT yet built/tested (needs a GPU claim to rebuild).
- Test harness staged: `~/host_cache.sh` — all arms `WP_ENSURE_BATCH_HOST=1`,
  delta = `WP_HOST_BUDGET_BYTES` (0/4G/8G) interleaved x2, NPRED=384. Measures
  `host_tier_hits`, NVMe GB, tok/s, coherence. Run order when card frees: scp
  edited wp-pager.cpp → rebuild build-hip → host_cache.sh.
- CAVEAT: HostTier free-lists are keyed by exact byte size; DS4's mixed 4/8-bit
  experts have variable page sizes → possible arena fragmentation → hit rate may
  trail the uniform-page sim. A Phase-0/1 measurement question, not a bug.

## 9. Non-goals / retained

- `WP_PREFETCH_XLAYER` (speculative cross-layer prefetch) stays default-OFF — net
  loss + a still-unfixed wrong-page corruption race (decisions 52669575 /
  7ce9ebd4). Not part of this design.
- p2p-direct NVMe→VRAM stays available but is NOT the default decode path here;
  its bandwidth advantage is moot (we're concurrency-bound) and it forfeits the
  RAM tier.
