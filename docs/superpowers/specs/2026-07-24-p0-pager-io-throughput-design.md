# P0 — Pager read throughput: honest instrumentation, transport choice, then the fix

**Date:** 2026-07-24
**Status:** design / pre-build.
**Roadmap:** `docs/dev/2026-07-24-distributed-paging-roadmap.md` (P0).
**Decision log:** `repo__kmbandy__llama.cpp` `cbb7417d`.

---

## 1. The gap

Measured 2026-07-24 with the pager's exact 4456448 B page, O_DIRECT, random
offsets, fresh files:

| drive | QD1 | QD4 | QD16 |
|---|---|---|---|
| main WD_BLACK SN850X 1000GB | 0.74–0.91 GB/s | 2.38–2.62 | 2.84–2.95 |
| 2026 WD Black SN750 250GB | 2.13–2.20 GB/s | 2.86–2.91 | 2.82–2.89 |

Production `io_effective_gb_s` in the same session: **1.514** (2-GPU, no RAM tier)
and **0.598** (4-GPU with RAM tier). I/O consumed ~59 s of a 78 s decode — **76% of
wall clock**. Headroom to the drive's own QD16 figure is roughly **2×**.

## 2. What the code actually does — and why the obvious framing was wrong

Established by reading `src/weight-pager/` (citations in §7):

1. **`WP_ENSURE_BATCH_HOST=1` bypasses P2P entirely.** The HOST O_DIRECT check
   sits *before* the `direct_to_device()` check in `ensure_batch` and returns
   unconditionally on success. Every recent run — and the standing `host_cache.sh`
   convention — therefore used a **pthread pool doing blocking `pread`**, not
   io_uring. A consequence worth stating: the HostTier-on-P2P integration
   committed earlier today (`6a1dcfe0d`) has **never executed** under these configs.

2. **The counters are mislabeled on that path.** `ensure_batch_submit_ms` is
   computed to a point *after* the completion wait, so it is essentially the whole
   NVMe read wall-clock; `ensure_batch_wait_ms` is `batch_seconds - read_seconds`,
   i.e. the H2D copy. The names were inherited from the P2P path, whose
   `Stats` comment still says "(P2P path)". **The observed "submit ≫ wait" is an
   artefact of the labels, not evidence about submission.**

3. **We are not at queue depth 1.** `avg_n 9.31` means ~9 concurrent `pread`s
   across a pool whose high-water mark is 16. The 2026-07-07 note that framed this
   as "QD1 serialized in `page_in_sync_`" predates this path and is obsolete.

4. **`ensure_batch` is a hard barrier.** It queues its whole miss set, blocks until
   every job reports done, then does H2D, then returns. Batch N+1's reads cannot
   overlap batch N's H2D or the intervening compute.

5. **All workers share one O_DIRECT fd per shard.** `pread` with an explicit offset
   is thread-safe, but whether the filesystem/driver takes an exclusive inode lock
   for concurrent O_DIRECT reads on a single fd is **not established from source**
   and needs a runtime trace.

6. **Prior art on the other path.** `wp-file-io.h` records that cold reads
   completing inline inside `io_uring_submit` serialised the submitter and capped
   random P2P at ~2 GB/s, fixed with `IOSQE_ASYNC` / `WP_IOWQ_MAX_WORKERS`. Same
   symptom signature, different transport — and that transport is currently unused.

**Therefore the first move is not "add concurrency."** At ~9 concurrent reads the
drive should already be near 2.8 GB/s and it is not, so the deficit is somewhere we
cannot currently see, because the instrument that would show us is mislabeled.

## 3. Goal and non-goals

**Goal:** raise sustained `io_effective_gb_s` toward the drive's demonstrated
~2.8–2.9 GB/s, on whichever transport proves better, without changing what the
pager reads or when.

**Non-goals.** No change to eviction policy, prefetch policy, HostTier semantics,
page sizing, or routing. No new speculation. Default behaviour unchanged until a
measurement justifies flipping a default.

## 4. Design — three units, strictly ordered

### Unit 1 — Honest instrumentation (no behaviour change)

The blocker for everything else. Requirements:

- Each transport reports its **own** phase breakdown under names that mean what
  they say: time building the job list, time computing alignment/bounce, time
  actually enqueueing, time blocked waiting for reads, time in H2D.
- One **transport-agnostic** headline metric — bytes actually read from storage
  divided by wall-clock spent in the read phase — comparable across HOST and P2P.
- Report **achieved concurrency**: how many reads were genuinely in flight, not
  how many were queued. `ensure_batch_max_n` and `avg_n` count queue occupancy,
  which is not the same thing.
- A one-line summary at teardown naming **which transport actually ran**. Nothing
  in the current logs states this plainly, which is how the P2P/HOST confusion
  survived.
- Carry forward the already-identified contamination fix: `ensure_batch_n_sub_sum`
  must not fold HostTier hits into the submission count, and `ensure_batch_max_n`
  must not be computed from `total_ok`.

Old field names may be kept as aliases if anything external reads them, but the
`Stats` comment claiming "(P2P path)" must stop applying to HOST-path numbers.

### Unit 2 — Transport A/B, decided by measurement

With Unit 1 in place, run the same workload across:
- HOST pthread pool (`WP_ENSURE_BATCH_HOST=1`), today's de facto default;
- P2P io_uring (`WP_ENSURE_BATCH_HOST=0`), including the `WP_IOWQ_MAX_WORKERS`
  setting that previously lifted it off ~2 GB/s;
- both with and without the RAM tier, since the submit/wait profile inverted
  between those two configs tonight and a single-config result would mislead.

Deliverable: a table of effective GB/s and honest phase breakdowns, and a
recommendation for the production default. **This is a measurement task, not a
code task**, and it requires a GPU claim.

### Unit 3 — The fix, scoped after Unit 2

Deliberately not specified yet, because choosing now would be guessing. Candidates
already visible, to be confirmed or eliminated by Unit 1/2 data:

- **Break the `ensure_batch` barrier** so batch N+1's reads overlap batch N's H2D
  and the intervening compute. This is the largest structural candidate: today the
  drive is idle during H2D and during compute.
- **Per-worker file descriptors** instead of one shared fd per shard, if a trace
  shows inode-lock contention.
- **Reduce bounce-buffer copies** from the 512-alignment path, if the prep/copy
  phase proves material once separately timed.
- **Adopt the P2P path** as default with its io-wq fix, if Unit 2 says it wins.

## 5. Verification

- **No GPU:** the counter arithmetic is pure and must be unit-tested in
  `tests/test-weight-pager.cpp` — given a synthetic sequence of phase timings and
  job outcomes, the reported concurrency, byte totals and effective GB/s must come
  out right, and HostTier hits must not inflate submission counts.
- **On hardware (needs a GPU claim):** decode must stay coherent and wikitext
  perplexity must not move beyond noise from the current baseline — this work must
  not change results, only speed. Effective GB/s and the phase table are the
  measurement.
- **Method:** interleave arms and alternate their order between rounds. Tonight's
  FFN-island A/B showed the arm that ran second was faster in all three rounds
  regardless of which arm it was; a fixed order would have reported a spurious win.

## 6. Risks

- Renaming or re-scoping counters invalidates comparisons with older logs. State
  the change loudly in the summary line so future sessions do not compare across
  the boundary. This is the same hazard as the 25.2 MiB/expert correction.
- The HOST path's 16 workers are a persistent, only-growing pool shared across
  calls; changing its lifecycle risks affecting load-time behaviour as well as
  decode. Keep Unit 1 strictly read-only with respect to behaviour.

## 7. Source references

`src/weight-pager/wp-pager.cpp` — `ensure_batch` HOST branch and its early return;
the phase counters printed every 1000 calls; `ensure_batch_submit_seconds` /
`ensure_batch_wait_seconds` assignment; `ensure_odirect_workers_ready_`,
`ensure_odirect_worker_loop_`, `ensure_odirect_fd_`, `ensure_odirect_worker_count_`.
`src/weight-pager/wp-file-io.cpp` / `.h` — `IoUringAsyncFileIO`, `create_file_io`
transport ladder, `set_iowq_max_workers` and the inline-completion comment.
`src/weight-pager/wp-pager.h` — the `Stats` block whose "(P2P path)" comment no
longer matches HOST-path usage.

---

## 8. Unit 2 results (2026-07-24, mad-lab-main, tip 547f94bc8)

DS4-Flash Q8, paging=ROCm0 (R9700), resident=ROCm1 (6900XT), slots 5500, `-c 4096`,
128 tokens greedy, 2 rounds with arm order alternated between rounds.
`llama-router.service` live throughout; port 8099, own-PID-only teardown.

| arm | transport | RAM tier | in-flight avg | NVMe GB | io_gb_s | tok/s | host hits |
|---|---|---|---|---|---|---|---|
| h0 | HOST O_DIRECT pool | off | 5.06 | **221.9** | 1.43 | **1.736** | 0 |
| h8 | HOST O_DIRECT pool | 8 GB | 5.28 | 158.9 | 0.74 | **0.960** | 5114 |
| p0 | SERIAL (buffered) | off | n/a | **82.7** | 1.59 | **1.911** | 0 |
| p8 | SERIAL (buffered) | 8 GB | n/a | 69.4 | 0.85 | 1.186 | 3457 |

All arms COHERENT. Per-arm samples: h0 1.694/1.778 · h8 0.905/1.015 ·
p0 1.809/2.013 · p8 0.782/1.590.

### 8.1 The HOST O_DIRECT path amplifies reads ~2.5x

h0 and p0 performed **identical work** — both logged exactly `page_ins: 20025` and
`evictions: 14541`. Useful payload is 20025 x 4456448 B = **89.24 GB**.

| path | NVMe bytes read | ratio to payload |
|---|---|---|
| HOST O_DIRECT pool | 221.9 GB | **2.49x** |
| SERIAL buffered | 82.7 GB | 0.93x |

Stable across rounds (222.67 / 221.15 and 82.94 / 82.48), so this is not noise.

**The drive was never the bottleneck.** 221.9 GB in 57.2 s of read-wait is
**3.88 GB/s** — *above* the 2.84-2.95 GB/s the same drive benchmarks at QD16. We
are not reading too slowly; we are reading roughly two and a half times more data
than we need.

Not the alignment bounce: that is "align-down prefix (<=511) + pad to 512", about
1.0002x. Cause not yet identified — that is the first job of Unit 3.

### 8.2 P2P does not exist on this configuration

With `WP_ENSURE_BATCH_HOST` unset the run reported
`transport=SERIAL (sync fallback)`, `p2p_b=0`, `serial_b=3310`.
`file_io_->direct_to_device()` is false, so the ladder falls past io_uring/P2P all
the way to the per-page sync path. Consequences:

- The "HOST vs P2P" question in §4 was never the real choice; the real comparison
  is **O_DIRECT worker pool vs buffered serial**.
- The `WP_IOWQ_MAX_WORKERS` candidate is moot until we know *why* P2P is
  unavailable.
- The HostTier-on-P2P integration in `6a1dcfe0d` still has never executed.

### 8.3 The RAM tier regressed throughput here — and the counters point at a
### defect in promotion, not at tiering

**READ THE CAVEAT FIRST.** n=2 per arm, and the serial+tier arm's two samples were
**0.782 and 1.590** — a 2x spread, wider than the effect being claimed. An earlier
revision of this section called the RAM tier "a net throughput loss"; that was an
over-claim from one underpowered run and has been retracted. Nothing below supports
a verdict on whether RAM tiering is worthwhile. What it does support is a specific
defect worth chasing.

**Bytes: the tier works.** NVMe falls 221.9 -> 158.9 GB (**-28%**, close to the
tiered design's predicted -24% at 8 GB) and 82.7 -> 69.4 GB (-16%).

**Time: the promotion path squanders the saving.** On h8 the new phase counters
show `read_wait` falling **57.2 -> 45.6 s (-11.6 s)** while `h2d` rises
**6.3 -> 35.3 s (+29 s)**. So:

| | |
|---|---|
| NVMe read time the tier SAVED | **-11.6 s** |
| H2D time the tier ADDED | **+29 s** |
| per-page promotion cost | ~29 s / 5114 hits = **~5.7 ms per 4.25 MB page** |
| implied promotion bandwidth | **~0.75 GB/s** on a link that should do ~25 GB/s |

~5.7 ms to move 4.25 MB is roughly **30x slower than the link allows**. That is a
synchronisation cost, not a bandwidth limit — the shape suggests promotion is
serial per page with a device synchronise, rather than batched or stream-ordered.

**The counterfactual is what matters.** At a plausible ~0.2 ms per page, 5114
promotions would cost ~1 s against an 11.6 s read saving — the tier would be a
clear net win of ~10 s. The byte reduction it delivers is genuinely valuable; the
current promotion implementation is giving it all back and more.

So the finding is: **promotion is ~30x off its own link speed, and fixing it is
what determines whether the tier's real byte savings can be realised.** That is a
bug to locate, and it needs proper replication plus a direct measurement of the
promotion path rather than inference from two runs.

**This still gates roadmap P2.** The distributed-paging direction assumes each
machine's RAM tier pays for itself. Until promotion is fixed and re-measured with
adequate replication, that premise is unverified in either direction.

### 8.4 The fastest arm is the one nobody chose

`SERIAL (buffered)` with no RAM tier is the best configuration measured: **1.911
tok/s**, +10% over the HOST pool, reading **2.7x fewer bytes**. Both its samples
exceed every HOST sample.

It is buffered — `dup_clear_o_direct` strips `O_DIRECT` for that path — so it
participates in the **OS page cache**, which `--no-mmap` plus O_DIRECT deliberately
bypasses. Its 0.93x ratio (below 1.0) is page-cache hits.

The uncomfortable reading: a custom RAM cache plus an O_DIRECT worker pool is
currently **losing to the kernel's own page cache** on a simple per-page read loop.

### 8.5 Consequences for Unit 3

Unit 3's targets are now evidence-based, and none of them is "add concurrency":

1. **Find and eliminate the 2.5x HOST read amplification.** Largest single effect;
   the drive is already delivering 3.9 GB/s.
2. **Fix the HostTier->VRAM promotion path** (~5.7 ms per page). Gates both P0 and
   the P2 split premise.
3. **Establish why `direct_to_device()` is false**, since P2P has silently not
   existed in any recent run.
4. **Re-evaluate the transport default.** Buffered serial currently wins; the
   page-cache interaction with `--no-mmap` deserves a deliberate decision rather
   than an inherited one.

Achieved concurrency measured 5.06-5.28 average against a peak of 16, so raising
sustained depth remains a *minor* lever compared with reading 2.5x less.

---

## 9. Post-fix validation (2026-07-24, tip a9f2c139b)

Same script, same settings, same corpus, HOST O_DIRECT path, no island, no RAM
tier — so the pre-fix PPL run is an exact control.

| | pre-fix | post-fix |
|---|---|---|
| `page_ins` | 264150 | **264150** (identical work) |
| `io_gb_read` (bytes the pager requested) | 1177.171 GB | **1177.171 GB** (identical) |
| **`ensure_batch_gb_s`** | **2.248** | **6.206** (**2.76x**) |
| wikitext PPL (8 chunks, n_ctx 512) | 1.9007 +/- 0.07421 | **1.9007 +/- 0.07421** |
| pread failures | **3** | **0** |
| wall | 818 s | **580 s** (-29%) |

**Correctness: identical to four decimals with the same stderr**, and every
per-chunk value matched ([5]2.1745, [6]2.0328, [7]1.9620, [8]1.9007). The fix
changes which byte *ranges* are read; it delivers the same payload.

**Amplification eliminated.** 1189.61 GB delivered for 1177.171 GB requested =
**1.011x**, against 2.49x measured pre-fix. The alignment resolved as intended:
`O_DIRECT alignment resolved to 4096 bytes (fstatfs ok, f_bsize=4096)` for every
shard.

**The EOF clamp did better than predicted.** §8 warned it might swap EIO for
EINVAL, since no shard size is 4096-aligned. Measured: **0 pread failures**,
down from 3. The clamp is genuinely handling the tail, not merely relocating the
error.

### 9.1 A ceiling correction

Earlier revisions of this spec quoted a "~2.84-2.95 GB/s drive ceiling at QD16"
and inferred "~2x available headroom". That figure came from a 128-read probe and
was **too low** — the SN850X is rated ~7 GB/s and this run sustains **6.206
GB/s**. Real headroom was materially larger than stated, which is why the
realised gain exceeds the predicted one. Treat short-burst probe numbers as a
floor, not a ceiling.

### 9.2 Not established by this run

- `ensure_batch_host_odirect_cap_skips` does not appear in the log at all, so it
  is unconfirmed whether it is zero or simply not printed. Follow-up.
- `inflight_avg_at_read_start` was **15.53** here versus 5.06 in the decode A/B.
  These are **not comparable** — prefill queues far deeper than decode. Nothing
  about achieved concurrency during *decode* is established by this run.
- Attribution between the two causes (alignment vs btrfs compression) is still
  unmeasured; both changed before this run.
- The RAM tier question is untouched here and remains open in both directions.
