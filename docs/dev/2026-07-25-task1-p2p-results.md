# 2026-07-25 results: HostTier zero-copy, P2P first execution, and a premise that did not hold

**Machine:** mad-lab-main · **Repo:** `~/GitHub/llama.cpp` master · **Tip at writing:** `04ecae824`
**Config of record:** unchanged from §4 of `docs/dev/2026-07-25-morning-pickup.md`
**Logs:** `~/wp_logs/task1/` (`summary.tsv`, per-arm `.log`), `~/wp_logs/task1_run.log`, `~/wp_logs/task1_run2.log`

## 1. Headline, stated at the confidence the data supports

The HostTier zero-copy promotion work landed and **does exactly what it was designed
to do**. It **did not** produce the throughput gain predicted for it, and the reason
it didn't is the most important finding of the day.

- `ensure_batch_host_jobs_ms` (where the per-hit 4.25 MiB memcpy lived):
  **1710 → 5.6 ms**, across four independent runs (5.3 / 5.6 / 5.8 / 4.9).
- `ensure_batch_host_zerocopy_promotions` **equals** `promotion_count` (3135) in every
  run — the borrow path is taken universally, no `lookup()` fallbacks, and
  `backend_pinned()` was true every time.
- Tier savings, reproducible: read_wait **15861 ± 43 ms → 13333 ± 97 ms (−2528 ms)**;
  NVMe **82.78 → 68.79 GB (−16.8%)**.
- Promotion H2D now costs nothing measurable versus no-tier: **3693 vs 3720 ms**, at
  **0.170 ms/page**. (For the record: the retracted "~5.7 ms/page, ~30× off link
  speed" figure was wrong by ~33×.)

**Decode throughput: unchanged.** n=4 per arm, round 1 dropped as cold:

| arm | values | mean | spread |
|---|---|---|---|
| h0 (no tier) | 3.088, 3.095, 3.096, 3.064 | 3.086 | 1.0% |
| h4 (4 GB tier) | 3.343, 3.072, 3.052, 3.091 | 3.140 | 9.4% |

Within-round deltas (adjacent arms share thermal/cache state, so this is the
comparison that survives any time trend): **+8.3%, −0.7%, −1.4%, +0.9% — median
+0.1%**. The +1.8% mean rests entirely on h4_2's 3.343, the single outlier.
**The tier is throughput-neutral.**

## 2. The premise that did not hold

The predicted gain (~+6%) came from arithmetic on the phase ledger: −2528 ms of
read_wait plus −1705 ms of memcpy, against a ~41 s generation. That arithmetic
assumed those milliseconds were wall-clock on the critical path.

**They are not.** Two independent multi-second reductions in pager phase time
produced no measurable wall-clock change. The direct reading is that these phases
are **already overlapped with compute**, so shrinking them cannot raise decode
throughput.

The supporting observation is sharper than the aggregate: **h4's tok/s varies 17%
(3.052–3.567) while every pager counter it owns stays constant** — read_wait
13221/13381/13397/13400, promo 3135 at ~530 ms, fresh 16890, pages 20025, NVMe
68.7–68.9 GB. Something outside the pager drives h4's throughput. A hypothesis not
yet tested: the 4 GB pinned `hipHostMalloc` arena on a 16 GB box with ~10 GB
available perturbs the host side of the token loop, which would explain why the
tier arm specifically is the unstable one while h0 sits at 3.086 ± 1.0%.

**Consequence for direction.** "The pager's read path is the decode bottleneck" —
the premise the P0 work was built on — is not supported at this configuration.
Before further read-path throughput work, establish where decode wall-clock
actually goes. Otherwise we risk optimising a path that is not on the critical
path, which is what happened today.

**What the change is still worth**, independent of throughput: 13.97 GB of memory
bandwidth no longer burned on the eval thread per run; a tier API that is correct
under concurrent writers, which host prefetch requires; and 16.8% fewer NVMe reads
at parity, i.e. drive endurance.

## 3. P2P executed for the first time

`~/wp_logs/task1/P4_p2p.log`:

```
wp::IoUringP2PFileIO: P2P enabled — pool dma_buf exported (23375.0 MiB VRAM),
  window cache max=64 page=4096 B (no full-pool host map, no host bounce)
wp::ensure_batch TRANSPORT: active=P2P (direct_to_device)
  host_batches=0 p2p_batches=3310 serial_batches=0
```

It was never broken. It was **never requested**: `create_file_io`
(`wp-file-io.cpp:786-796`) gates the whole ladder behind `LLAMA_WP_TRANSPORT=p2p`,
which no harness had ever set. Reaching it *also* requires `WP_ENSURE_BATCH_HOST`
unset, because the HOST O_DIRECT path is checked first in `ensure_batch` and returns
unconditionally.

Transports at a 4 GB tier, identical deterministic workload (20025 pages every arm):

| transport | tok/s | read GB/s | NVMe read |
|---|---|---|---|
| HOST O_DIRECT pool | 3.140 | 4.39 | 68.7 GB |
| P2P direct-to-device | 2.312 | 2.92 | **67.78 GB** |
| SERIAL sync fallback | 1.958 | 1.70 | 72.33 GB |

P2P has the **lowest** read amplification of any arm, and no H2D stage at all.
kmbandy's prior measurement puts the BAR/SAM write path at **≥6 GB/s**, above HOST's
achieved 4.39 — so P2P's 2.92 GB/s is a software limit with real headroom, and a
hardware-ceiling explanation is ruled out. Do not re-derive this.

Two defects located in source, one already quantified:

- **Tier promotion on P2P was serial and still copying** — `page_in_sync_`, 999.5 ms
  vs HOST's 530 ms for the same 3135 pages. Fixed in `04ecae824` by extending
  `borrow()`/`release()` to that path.
- **A tier enabled always disabled direct-to-device in `page_in_sync_`**
  (`host_store_possible` is always true, since every expert page fits the budget), so
  tier-eligible pages gave up the one thing P2P exists for. Now selectable via
  `WP_P2P_DIRECT_TO_DEVICE`, defaulting to the old behaviour.

Instrumentation was the prerequisite and is now in: the `ensure_batch_host_*` phase
counters increment only inside the HOST block, and the in-flight tracker is the
O_DIRECT pool's, so **P2P's reported `inflight avg=0.00` and `0.0ms` phases were
absence of instrumentation, not absence of time.** P2P's real concurrency remains
unmeasured until the first instrumented run.

## 4. Corrections to the record

1. **`unit3b_ab.sh`'s `p0`/`p4` arms were never P2P.** They only unset
   `WP_ENSURE_BATCH_HOST` and never set `LLAMA_WP_TRANSPORT`. Every number those arms
   ever produced — including last night's `p0 2.287` / `p4 1.913` — is host-ladder or
   serial, not P2P. The arm names were misleading in every log they wrote.
2. **My own first P2P probe arm was invalid** for the same class of reason: it set
   `WP_ENSURE_BATCH_HOST=1` alongside `LLAMA_WP_TRANSPORT=p2p`, so the HOST path
   preempted P2P and it silently re-measured h4 (its counters match h4_3 to 0.3%).
3. **Last night's h0 = 3.570 was not a stable baseline.** Today's h0 is 3.086 ± 1.0%
   over four runs. 3.570 was a lucky first run of the same kind as today's dropped
   h0_1 (3.477). Cross-session tok/s comparisons in either direction — including
   reading 3.570 → 3.140 as a regression — are not supportable.
4. **The predicted "~+6% decode" from zero-copy did not occur.** See §2.
5. **The window-granularity hypothesis was wrong.** The `page=4096 B` in the P2P
   startup line is mmap *alignment* granularity, not window size:
   `wp-file-io-p2p.cpp:516-519` aligns the offset down and rounds the length up, so
   one window maps a whole ~4.25 MB request. There is no "many mappings per page"
   problem.

## 5. The model is essentially incompressible (bears on the split)

A compressed copy of all four shards (`~/models/ds4_zstd`, built for the pending
amplification attribution) reproduces the original distribution and localises it:

| shard | size | encoded extents | total | compressed |
|---|---|---|---|---|
| 1 | 43.6 GB | 111 | 2519 | 4.4% |
| 2 | 43.6 GB | **0** | 1542 | 0% |
| 3 | 43.6 GB | **0** | 6861 | 0% |
| 4 | 18.3 GB | **1747** | 1923 | **91%** |

Q8 weights do not compress; compressibility is concentrated almost entirely in
shard 4, ~12% of total bytes.

**This makes compression an implausible cause of the 2.49× amplification.** Shards 2
and 3 (88 GB, 8403 extents) contain zero compressed extents and still served reads,
yet amplification measured a uniform 2.49× across two rounds (222.67 / 221.15 GB).
Alignment is the far more likely cause. **Still unmeasured**, and the clean
single-variable test is not compressed-vs-uncompressed but forcing alignment back to
512 on the *uncompressed* file — which needs a debug-only override, since
`resolve_odirect_alignment()` deliberately takes `statfs f_bsize` as the sole
authority.

For the distributed split this is close to free: if the model barely compresses,
storing it uncompressed on mad-lab-2026 costs no disk, so "store models
uncompressed" stops being a tradeoff. Worth confirming against the actual shard
placed there.

Incidental: shard 4 being 91% compressible is unusual for a weights file and
suggests it is largely zeros or highly redundant (possibly the MTP head or
padding). If it is largely inert it may be cheap to place anywhere.

## 6. Commits

| commit | what |
|---|---|
| `960f3bcda` | HostTier zero-copy borrow/release **spec** |
| `b8b7f74da` | borrow/release refcounting, eviction-skip, deferred retirement, RAII guard, 8 tests |
| `e3adcb91a` | generation handles — closes the same-page_idx release ambiguity |
| `f129aeda7` | P2P instrument-and-tune **spec** |
| `04ecae824` | P2P phase/concurrency counters, tunables, `page_in_sync_` zero-copy |

New knobs (all defaulting to prior behaviour): `WP_P2P_QUEUE_DEPTH`,
`WP_P2P_WINDOW_CACHE_MAX`, `WP_P2P_DIRECT_TO_DEVICE`; `WP_IOWQ_MAX_WORKERS` now
logged at P2P startup.

## 7. Open, in priority order

1. **Where does decode wall-clock actually go?** §2 makes this the gating question
   for any further read-path work.
2. **Amplification attribution** — needs the debug alignment override; the compressed
   copy at `~/models/ds4_zstd` exists and should be **deleted** after use.
3. **P2P tuning sweep** — now instrumented and tunable; unmeasured.
4. **The distributed split** (roadmap P2) — the stated first priority, untouched
   today. Blocker remains structural: llama.cpp has no pipeline parallelism and RPC
   cannot express it. 8 GB RAM tier per machine is settled; do not re-litigate.
