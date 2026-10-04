# Pipelined tier promotion: overlap RAM→VRAM with NVMe→VRAM — design

**Date:** 2026-07-25
**Repo:** `~/GitHub/llama.cpp` on mad-lab-main, master, from tip `39a452ee8`
**Predecessors:** `docs/superpowers/specs/2026-07-25-hosttier-zerocopy-promotion-design.md`,
`docs/superpowers/specs/2026-07-25-p2p-read-path-instrument-and-tune-design.md`
**Measured context:** `docs/dev/2026-07-25-p2p-tuning-results.md`

## 1. The architecture this serves

kmbandy's design for mad-lab-main: **two independent paths into VRAM, running at the
same time.**

- **Warm pages** (prefetched or evicted-victim) come from the RAM tier: RAM→VRAM H2D.
- **Cold misses** come straight off NVMe into the VRAM slot: NVMe→VRAM peer DMA,
  touching host RAM not at all.

The payoff is not primarily aggregate bandwidth. It is that miss traffic stops
consuming host RAM: no bounce buffer, no D2H staging, no capacity spent on pages in
transit, and no 2x RAM-bandwidth cost per miss byte (today a miss is written to RAM
then read back out of it). On an 8 GB-per-machine budget that capacity matters more
than the throughput does. A secondary and sharper benefit: cold misses stop
*polluting* the tier, so the tier holds pages the prefetcher deliberately chose
rather than whatever was missed most recently.

The two routes are genuinely independent hardware paths — a HIP transfer stream
versus NVMe peer DMA into the dma_buf-exported BAR — sharing only the GPU's x16
link, which has headroom for both.

## 2. What already exists, and what the actual defect is

**The dual path is already implemented and already correct.** In `ensure_batch`'s P2P
branch (`wp-pager.cpp:2710+`): tier hits are consulted first, and cold misses go into
`submit_batch(reqs)` with `slot_ptr_(cold[k].slot)` as the destination — NVMe direct
to the VRAM slot at true QD=N. No transport plumbing needs changing. No second
`FileIOLayer` is needed (and note the pager already holds two —
`file_io_` and `host_prefetch_file_io_`, `wp-pager.h:621-622`).

**The defect is ordering.** Tier hits are promoted **serially, one `page_in_sync_`
call at a time, in a loop that runs to completion before the cold-miss batch is
submitted.** So the NVMe reads — the long pole at ~8 ms each — cannot start until
every RAM promotion has finished. The two paths are used *sequentially*.

**The blocking is host-side, not device-side.** `GpuTransport::stage_in()` is
`stage_in_async()` followed by `hipStreamSynchronize()` (`wp-gpu-transport.cpp:121-141`).
The async primitive already exists, already enqueues `hipMemcpyAsync` +
`hipMemsetAsync` on the transport's own stream, and already records a completion event
without synchronizing. The file says so explicitly at lines 190-196: *"Record the
completion event on the transport stream BEFORE we synchronize so a future async-aware
caller can hipStreamWaitEvent on it from another stream... Phase 1e can flip this to
truly pipelined behaviour without changing this function's signature."*

This spec is that flip.

**Why it matters more later than now.** At today's 15.7% hit rate the serialization
costs ~561 ms per run. It scales **linearly with hit rate** — which is exactly what
prefetch exists to raise. At a 50% hit rate it would be ~1.8 s of RAM promotion
blocking ahead of every NVMe submission. Prefetch would be partly self-defeating
until this is fixed, so this is a prerequisite for the prefetch work, not an
optimisation of it.

## 3. Scope: general, not P2P-only

Make the batched-async promotion the general mechanism for tier→VRAM promotion, used
by every path that promotes, rather than a special case bolted into the P2P branch:

- **`ensure_batch` P2P branch** — the motivating case. Reorder so the cold-miss batch
  is submitted first, tier promotions are enqueued async while those reads are in
  flight, then P2P completions are reaped, then one synchronisation covers the
  promotions.
- **`ensure_batch` HOST branch** — already batches its H2D and already ends in a
  single `hipDeviceSynchronize()`. It should use the same promotion helper rather than
  keeping a parallel implementation. Do not regress its existing per-group event
  timing (`h2d_ev_start/mid/end`) or its promotions-then-fresh enqueue order.
- **`page_in_sync_`** — retains a synchronous entry point, because callers depend on
  the "data is in VRAM on return" contract. Implement that entry point in terms of the
  async helper plus an immediate wait, so there is one promotion implementation rather
  than two.
- **`wp-prefetch.cpp`** — also calls `stage_in`/`stage_in_async` (lines ~343, ~201-270).
  Bring it onto the shared helper only if it is a clean fit; if its lifetime model
  differs, leave it and say so. Do not force it.

A single helper that takes a set of (page, slot) promotions, borrows each from the
tier, enqueues the H2D async, and returns something the caller synchronises once — with
a synchronous convenience wrapper for one page — is the shape to aim for. The exact
signature is the implementer's call.

## 4. Correctness constraints — these are the whole risk

### 4a. The compute stream must not read a slot before its promotion completes

This is a **display-wedging, silent-wrong-weights bug class with history**, not a
throughput concern. `wp-gpu-transport.cpp:166-178` records why promotions were moved
off the default stream: synchronous `hipMemcpy` there *"does NOT auto-serialize with
GGML's non-blocking compute stream (common.cuh:1439). That created a torn-write race
against MMQ kernels reading the same slot AND contributed to compute/graphics ring
scheduling pressure that wedged the display GPU under MoE-decode load."*

Today the host-blocking sync inside `stage_in()` is what upholds the ordering. Removing
it per-page means the guarantee must be re-established explicitly. Either is
acceptable, and the choice should be stated in the commit message with its reasoning:

- one stream synchronisation before `ensure_batch` returns (what the HOST branch
  already does), or
- a device-side `hipStreamWaitEvent` making the compute stream wait on the transport
  event.

**Whichever is chosen, no `ensure_batch` exit path may return with a promotion still
in flight.** The R9700 also drives the desktop displays, so getting this wrong is
visible to the user, not just to a benchmark.

### 4b. Borrow lifetime extends to the sync, not to the enqueue

Promotions now source their H2D directly from the pinned HostTier arena via
`borrow()`. With async enqueue, the borrow must stay held until the transfer is
**observably complete** — i.e. past the synchronisation, not past the enqueue.
Releasing at enqueue time reopens exactly the wrong-expert-weights hole that
`borrow()`/`release()` was built to close.

Use the generation-handle form (`e3adcb91a`) and a scope guard owning the borrow list,
as the existing call sites do. Note the existing `page_in_sync_` guard is safe *because*
`stage_in` synchronises internally; once it doesn't, that guard's placement must be
re-derived rather than assumed still correct.

### 4c. Event pool exhaustion is a real failure mode, not theoretical

`stage_in_async` returns -1 and warns *"event pool exhausted (queue depth too small?)"*
when `free_events_` is empty. The pool is sized at init:
`prefetch_depth*2+8` when async-ensure is on, else `prefetch_depth+2`
(`wp-pager.cpp:966-968`) — 40 or 18 today at `prefetch_depth=16`.

Batching promotions holds several events outstanding simultaneously, and the entire
premise of this work is that prefetch will raise the number of hits per batch. Handle
it deliberately: either bound in-flight promotions and drain when the bound is hit, or
size the pool from the maximum batch size. **Exhaustion must not silently degrade to a
dropped promotion** — a page that fails to promote must fall through to a real read,
and the event must be counted so a log shows it happened.

### 4d. Ordering within a batch

Cold-miss reads write into VRAM slots; promotions write into different VRAM slots. They
must not target the same slot in one batch. The existing code assigns each miss its own
slot before either path runs, so this holds today — verify it still holds after the
reorder rather than assuming.

## 5. Measurement

Add a counter for promotions that were enqueued async and overlapped, distinct from
those that took a synchronous path, plus a counter for event-pool exhaustion events.
Both must print in the summary. `ensure_batch_host_odirect_cap_skips` is the standing
example of a counter that existed and never printed, and it cost real time.

The expected effect is that P2P `read_wait` absorbs the promotion time rather than the
two adding. Note explicitly for whoever measures this: **four read-path improvements on
2026-07-25 produced no tok/s change**, because in-flight depth is pinned at 5.29 by
~6.05 pages per `ensure_batch` call. Expect this change to show up in phase counters
and in prefetch headroom, **not** necessarily in decode throughput. Do not report a
tok/s win from a single arm; the tuned P2P arms spanned 2.219-3.093 while the untuned
baseline was reproducible to 0.1%.

## 6. Out of scope

- The full-pool host map for P2P (separate, evidenced, ~15% bandwidth).
- Raising pages-per-batch (prefetch / MTP) — the only route past the 5.29 ceiling, and
  the thing this change unblocks rather than does.
- Any tuning-knob default changes.

## 7. Tests

`tests/test-weight-pager.cpp`, existing conventions exactly: `static int test_*()`
returning a failure count, `EXPECT`/`EXPECT_EQ_INT`, registered in the table in
`main()`. HostTier lifetime and bookkeeping are testable CPU-only.

Required: borrow is still held across a deferred completion; a promotion whose event
acquisition fails falls through rather than silently dropping; the event-accounting
bound behaves at and past its limit; and the existing HostTier borrow/release suite
still passes unchanged.

## 8. Build and verification

Build `build-hip` and report the real exit status. The library target is
`llama-common`; **there is no target called `common`** — asking for one exits 0 having
done nothing. Confirm from object timestamps that the files you changed actually
recompiled; a target that skips them still exits 0, and that produced a false pass
earlier today.

**No GPU workload, no inference, no benchmarks.** No board claim is held. Correctness
and compilation only; the overlap measurement is a separate step under a claim.

## 9. Shared-tree rules — non-negotiable

kmbandy's uncommitted work is in this tree, including `docs/examples/router-fleet-main.ini`.
Never run `git checkout`, `git restore`, `git stash`, `git reset`, `git add -A`, or
`git commit -a`. Stage only files you touch, by explicit path. If a file you need
already has someone else's uncommitted changes, stop and report instead of committing
around it. A live `llama-router.service` is running: never `pkill`/`pgrep` by pattern.

Expected files: `src/weight-pager/wp-pager.{h,cpp}`,
`src/weight-pager/wp-gpu-transport.{h,cpp}`, possibly
`src/weight-pager/wp-host-tier.h`, `tests/test-weight-pager.cpp`.
