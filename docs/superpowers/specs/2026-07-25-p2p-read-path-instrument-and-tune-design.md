# P2P read path: instrument, then close the gap to HOST — design

**Date:** 2026-07-25
**Repo:** `~/GitHub/llama.cpp` on mad-lab-main, master, from tip `e3adcb91a`
**Predecessor:** `docs/superpowers/specs/2026-07-25-hosttier-zerocopy-promotion-design.md`

## 1. What changed this morning

P2P direct-to-VRAM reads **executed for the first time** on this machine
(`~/wp_logs/task1/P4_p2p.log`):

```
wp::IoUringP2PFileIO: P2P enabled — pool dma_buf exported (23375.0 MiB VRAM),
  window cache max=64 page=4096 B (no full-pool host map, no host bounce)
wp::ensure_batch TRANSPORT: active=P2P (direct_to_device)
  host_batches=0 p2p_batches=3310 serial_batches=0
```

It had never run before, and not because it was broken. `create_file_io`
(`wp-file-io.cpp:786-796`) gates the entire P2P ladder behind
`LLAMA_WP_TRANSPORT=p2p`, which no harness had ever set. Reaching it *also*
requires `WP_ENSURE_BATCH_HOST` unset, because the HOST O_DIRECT path is checked
first in `ensure_batch` and returns unconditionally — a first probe arm that set
both silently re-measured the HOST path with an idle P2P layer beneath it.

Measured, 4 GB tier, identical deterministic workload (20025 pages every arm):

| transport | tok/s | read GB/s | NVMe read |
|---|---|---|---|
| HOST O_DIRECT pool | 3.140 | 4.39 | 68.7 GB |
| **P2P direct-to-device** | **2.312** | **2.92** | **67.78 GB** |
| SERIAL sync fallback | 1.958 | 1.70 | 72.33 GB |

## 2. Why P2P should end up FASTER than HOST, not merely equal

Three independent reasons:

1. **It deletes a whole stage.** HOST reads into a pinned bounce buffer and then
   does an H2D copy — ~3.7 s per run in `ensure_batch_host_h2d_ms`. P2P lands
   bytes in the VRAM slot directly: no bounce, no H2D.
2. **Its read amplification is already the best of any arm** (67.78 GB, below
   HOST's 68.7 and serial's 72.33). Whatever limits P2P, it is not reading too
   much.
3. **The hardware ceiling is not the limit.** kmbandy has previously measured the
   BAR/SAM write path at **≥6 GB/s** — above the 4.39 GB/s HOST currently
   achieves. So P2P's 2.92 GB/s is a software limit with real headroom, and a
   hardware-ceiling explanation is ruled out. Do not re-derive this; it is
   settled.

The goal of this work is therefore not parity. It is to find out why a path with
strictly less work to do is running at two-thirds the bandwidth.

P2P also matters for the distributed roadmap (P2) independently of speed: it
needs no host bounce buffers, and the second machine's budget is 8 GB of RAM
total. Even at today's 2.92 GB/s it may be the correct transport there.

## 3. Two defects already located in source (no new experiment needed)

### 3a. Tier promotion on the P2P path is serial AND still copying

Measured, same 3135 pages in both arms:

| arm | tier promotions | promo time | zerocopy counter |
|---|---|---|---|
| h4 (HOST) | 3135 | 530 ms | 3135 |
| P4_p2p | 3135 | **999.5 ms** | **0** |

1.89× the time for identical work. `promo n=3135(sync)` means promotions ran
through `page_in_sync_`, which is per-page serial and still calls the copying
`HostTier::lookup()` (`wp-pager.cpp:4167`). This morning's zero-copy work
deliberately scoped only the HOST batch path, on the assumption `page_in_sync_`
was a cold fallback. **On P2P it is not a fallback — it is the promotion path.**
So P2P pays both the serialization and the full-page memcpy that was just
removed everywhere else. Roughly 470 ms of the gap, already quantified.

The `borrow()`/`release()` API from the predecessor spec is the fix; it was built
to be reusable and needs no redesign. The lifetime rules there apply unchanged:
release only after the transfer is known complete, release on every exit path via
a scope guard, use the generation handle, and fall back to `lookup()` when
`backend_pinned()` is false. `page_in_sync_` uses `transport_.stage_in(...)`
followed by `release_event(evt)` rather than a batch-wide
`hipDeviceSynchronize()`, so the correct release point must be established from
what `stage_in`/`release_event` actually guarantee about completion — read them
rather than assuming, and if `stage_in` is asynchronous the release belongs after
whatever call makes it observably done.

### 3b. Enabling the RAM tier silently disables direct-to-device in `page_in_sync_`

`wp-pager.cpp:4268-4270`:

```
const bool host_store_possible = host_tier_ && m.size <= host_tier_->budget_bytes();
bool direct_to_device = file_io_->direct_to_device() && !host_store_possible;
```

Every expert page is smaller than the tier budget, so with a tier enabled
`host_store_possible` is **always** true and `direct_to_device` is **always**
false on this path. The read is routed into staging so the bytes can also be
stored to the tier. The intent is legible — populate the victim tier on the way
past — but the cost is that tier-eligible pages give up direct-to-VRAM entirely,
which is the one thing P2P exists to do.

This is a genuine design tension, not an obvious bug, and it should be resolved
by measurement rather than by picking a side in the spec. Make the behavior
selectable so both can be measured: direct-to-VRAM-and-skip-the-tier-store versus
today's read-into-staging-and-store. Default to today's behavior so the change is
inert until measured. Report which one each run used.

## 4. Phase 1 (do this first): make the P2P path observable

The P2P path is currently being optimized blind, and every wrong call made
yesterday and today came from reasoning on counters that did not cover the path
in question. Specifically:

- `ensure_batch_host_*` phase counters are incremented **only** inside the HOST
  block (`wp-pager.cpp:2653-2656`), so P2P reports `jobs=0.0ms read_wait=0.0ms
  h2d=0.0ms` — absence of instrumentation, not absence of time.
- The in-flight tracker is `ensure_odirect_inflight_`, the O_DIRECT worker pool's.
  P2P's `inflight avg=0.00 peak=0` therefore says nothing about P2P's real queue
  depth. **P2P's achieved concurrency is currently unknown.**
- `ensure_batch_host_fresh_count` is HOST-only and `page_in_sync_fresh_count` was
  0, so the P2P arm's fresh reads are counted by nothing at all.

Required: a phase/concurrency breakdown for the P2P batch path that answers, per
run, where its wall-clock goes and how many reads it actually keeps in flight.
Mirror the HOST path's existing counter shape and naming so the two are directly
comparable arm-to-arm — that comparability is the point. Every new counter must
appear in the printed summary; `ensure_batch_host_odirect_cap_skips` is a live
example of a counter that exists and never printed, and it cost a session's worth
of blind spots.

Phase 1 is a prerequisite, not a nicety. Do not tune anything before it lands.

## 5. Phase 2: tunables to expose (measurement comes after)

Correcting a hypothesis I had wrong before reading the code: the
`page=4096 B` in the startup line is the mmap **alignment granularity**, not the
window size. `wp-file-io-p2p.cpp:516-519` aligns the offset down and rounds the
length up, so one window maps a whole ~4.25 MB request. There is no
"1088 mappings per page" problem. Do not go looking for one.

The real knob is the window cache depth, `wp-file-io-p2p.cpp:418-423`:

```
max_windows_ = queue_depth * 4;   // then clamped to [64, 256]
```

With `WP_IOURING_DEPTH=16` that pins `max_windows_` to its 64 floor, and it is
derived from queue depth rather than set independently. Expose as env-tunable,
with today's derivation as the default so nothing changes until deliberately
swept:

- the window cache cap, independently of queue depth
- the P2P queue depth, if it is not already honoring `WP_IOURING_DEPTH` on this
  path — verify which, and say so
- `WP_IOWQ_MAX_WORKERS`, which reaches live code for the first time now that P2P
  actually runs and has never been exercised

Log the resolved value of each at startup next to the existing "P2P enabled"
line, so a log says what a run used instead of what it defaulted to.

## 6. Out of scope

Batching `page_in_sync_` — turning per-page serial promotion into a batched
submit — is a larger change with its own correctness surface. Note it as
follow-up work; do not attempt it here.

No throughput conclusions belong in this work. Phase 1 and 2 make the path
measurable and tunable; the sweep and the verdict are a separate step run under a
GPU claim.

## 7. Verification

Unit-testable pieces (counter plumbing, env parsing and clamping, the
`page_in_sync_` borrow lifetime) go in `tests/test-weight-pager.cpp` in the
existing style: `static int test_*()` returning a failure count, `EXPECT` /
`EXPECT_EQ_INT`, registered in the table in `main()`.

Build `build-hip` and report the real exit status. The library target is
`llama-common`; **there is no target called `common`** — asking for one exits 0
having done nothing. The P2P code is inside `#if defined(LLAMA_HAVE_IO_URING)`
and HIP guards; confirm from object timestamps that the files actually
recompiled, because a target that skips them still exits 0.

**Do not run any GPU workload, inference, or benchmark.** No board claim is held.
Correctness and compilation only.

## 8. Shared-tree rules — non-negotiable

kmbandy's uncommitted work is in this tree. Never run `git checkout`,
`git restore`, `git stash`, `git reset`, `git add -A`, or `git commit -a`. Stage
only the files you touch, by explicit path. `git status` shows many modified
files that are not yours — leave every one alone. If a file you need already has
someone else's uncommitted changes, stop and report rather than committing around
it. A live `llama-router.service` is running: never `pkill`/`pgrep` by pattern.

Expected files: `src/weight-pager/wp-file-io-p2p.cpp`,
`src/weight-pager/wp-pager.{h,cpp}`, possibly `src/weight-pager/wp-file-io.h`,
and `tests/test-weight-pager.cpp`.
