# HostTier zero-copy promotion — design + build spec

**Date:** 2026-07-25
**Task:** Task 1 of `docs/dev/2026-07-25-morning-pickup.md`
**Repo:** `~/GitHub/llama.cpp` on mad-lab-main, master, from tip `9537874a9`

## 1. The problem, measured

`HostTier::lookup()` (`src/weight-pager/wp-host-tier.cpp:204-216`) copies a full
page out of the tier arena into a caller-owned bounce buffer, holding `mu_` for
the duration. On the HOST `ensure_batch` path that is one 4456448-byte
(4.25 MiB) RAM→RAM `memcpy` per tier hit, on the eval thread, in the critical
path of the token.

Measured on the config of record (DS4-Flash, 4 GB tier, one run):

| | |
|---|---|
| tier hits per run | 5114 |
| bytes copied | 13.97 GB |
| time | 1.71 s (≈8.2 GB/s — single-threaded `memcpy` speed) |
| reproducibility | 1.2% across runs |

That time is attributed today by `ensure_batch_host_jobs_seconds` (the
`io_t0 → tp_jobs` phase at `wp-pager.cpp:2261`), because the lookup loop is the
only substantial work in that phase. **This is the counter that verifies the
fix:** it should collapse to near zero.

Current tier ledger, 4 GB arm vs no tier: `read_wait` −2387 ms, this memcpy
+1710 ms, `h2d` +187 ms — net **−490 ms**, i.e. a wash. Removing the copy takes
the ledger to roughly **−2200 ms (~+6% decode)** and is the difference between
the RAM tier paying for itself and not.

The copy is redundant on its face: the arena is already `hipHostMalloc` pinned
memory (`init()`, `backend_pinned_`), so it is a legal, page-locked source for
`hipMemcpyAsync` directly. The bounce buffer adds a hop that the DMA engine
does not need.

## 2. Why this is not a deletion

The copy is load-bearing for a reason stated in the header comment at
`wp-host-tier.h:31-33`: because the bytes are consumed *inside* `mu_`, a
concurrent `store()` / `store_from_device()` / `erase()` cannot reclaim the
arena slot before the caller is done with it. The HOST path relies on that — it
enqueues every H2D and only commits at a single `hipDeviceSynchronize()`
(`wp-pager.cpp:2464`), so with a zero-copy source the arena slot must stay
valid from enqueue until that sync returns.

The race is real, not theoretical. `HostPrefetcher` is a started background
thread (`wp-pager.cpp:1080-1092`) whose store callback is
`host_tier_->store(page_idx, bytes, n)`. `store()` → `acquire_slot_()` →
`evict_one_lru_()` can hand our still-in-flight offset to a different page and
`memcpy` over it. `store_from_device()` (the victim path, `wp-pager.cpp:1283`)
can do the same from the eval thread. Under today's config of record
(`WP_HOST_PREFETCH` off) the prefetch thread does not run, so the window is
currently closed — but prefetch is the next feature in line, and a
"works because nobody is writing" invariant is not one to ship.

Failure mode if we get this wrong: a page of *some other expert's* weights
lands in a VRAM slot committed as this expert. Silently wrong logits, no crash.
This is the wrong-expert-weights class the existing sync-failure guard at
`wp-pager.cpp:2461-2464` was written to prevent, and it must not be reopened.

## 3. Chosen approach: borrow / release with a per-entry refcount

Rejected alternatives, briefly:

- **Pin borrowed pages permanently / never evict.** Leaks tier capacity
  monotonically; the whole point of the arena is churn.
- **A single "in use" boolean per entry.** Breaks if one page appears twice in
  a batch's miss list, and gives no safe way to nest the serial fallback path
  under a batch. A count costs the same and has neither problem.
- **Copy on a worker thread instead of the eval thread.** Still burns 13.97 GB
  of memory bandwidth and adds a synchronization point; treats the symptom.

The design:

**`borrow(page_idx, &src_out, n)`** — under `mu_`, resolve the resident entry,
require `bytes == n` exactly as `lookup()` does, touch the LRU, increment that
entry's borrow count, and hand back `arena_ + offset`. No copy. Returns false
on any miss, in which case `src_out` is untouched and the caller must fall
through to a fresh read exactly as a `lookup()` miss does today.

**`release(page_idx)`** — under `mu_`, decrement the borrow count. If it
reaches zero and the entry was retired while borrowed, complete the deferred
reclamation now.

**Eviction respects borrows.** `evict_one_lru_()` must not reclaim a borrowed
entry. It walks from the LRU front to the first entry with a zero borrow count
and evicts that one; if every resident entry is borrowed it returns false, and
`acquire_slot_()` already handles that by warning and failing the store
(`wp-host-tier.cpp:272-276`). A failed soft-prefetch store is a
non-event — it is soft by construction.

**Retirement of a borrowed entry is deferred, not blocked.** `erase()`, and the
implicit erase inside `store()` / `store_from_device()`, must remove the page
from `resident_` and the LRU immediately — so `contains()` goes false, the
RAM/VRAM exclusivity invariant holds, and a re-store of the same page gets a
*different* slot with no aliasing — while withholding the offset from the free
list until the borrow count drains. Pending entries therefore need to be held
somewhere other than `resident_`, keyed so `release()` can find them.

**`lookup()` stays.** It keeps its current copy semantics and its current
callers. Only the HOST batch path converts to borrow/release. The serial
`page_in_sync_` path (`wp-pager.cpp:4166`) is the fallback, is not hot when the
HOST path is enabled, and converting it buys nothing while adding a second
lifetime to reason about. Out of scope.

**Pinning guard.** Take the zero-copy path only when
`host_tier_->backend_pinned()` is true. If `hipHostMalloc` failed and the arena
is plain `malloc`, an H2D from it is unpinned — HIP stages it internally and the
copy stops being async, which could be *slower* than today's
memcpy-into-pinned-bounce. In that case use `lookup()` as now. The bounce
buffers must therefore keep existing and keep being allocated; this change does
not remove them.

## 4. Call-site changes (HOST batch path, `wp-pager.cpp` ~2229-2470)

The eval thread's sequence becomes: borrow what the tier has, read the rest,
enqueue all H2Ds — promotion copies sourced straight from the arena, fresh
copies from their bounce buffers as now — one `hipDeviceSynchronize()`, then
release every borrow.

Requirements on that code:

- The promotion H2D source becomes the borrowed arena pointer. Fresh-read
  sources are unchanged, including the `+ jobs[k].prefix` O_DIRECT offset.
- **Every** exit from the region between borrow and sync must release. That
  includes the cap-skip path, the enqueue-failure path
  (`issue_h2d_copy` clearing `host_hit[k]`), the `sync_ok == false` path that
  forces the whole batch to the fallback, and any early return. Do this with a
  scope guard that owns the borrowed-page list rather than by auditing exits by
  hand — the auditing approach is how this class of bug ships.
- Release must happen *after* `hipDeviceSynchronize()` returns, never before.
  The existing `host_tier_->erase(mm.page)` on successful commit
  (`wp-pager.cpp:2516`) can stay where it is; with deferred retirement, erase
  before release is safe and order-independent.
- Because the batch may fall back to `page_in_sync_`, which calls `lookup()` on
  the same pages, releases must have completed before the fallback runs — or,
  equivalently, the entries must still be resident. Only-on-success `erase()`
  already gives that.
- The existing per-group H2D event timing (`h2d_ev_start/mid/end`) and the
  promotion/fresh counters keep working unchanged; enqueue order stays
  promotions-then-fresh.
- Add a counter for zero-copy promotions actually taken, so a future session
  can tell the fast path from the `lookup()` fallback in a log instead of
  inferring it. It belongs next to the existing
  `ensure_batch_host_promotion_h2d_seconds` reporting and must appear in the
  printed summary — `ensure_batch_host_odirect_cap_skips` is a live example of a
  counter that exists but never prints, which is worth not repeating.

## 5. Tests (`tests/test-weight-pager.cpp`)

Follow the file's existing conventions exactly: `static int test_*()` returning
a failure count, `EXPECT` / `EXPECT_EQ_INT`, registered in the table in
`main()`. These are CPU-only — `HostTier` on a non-HIP build uses `malloc`, so
all of the lifetime logic is testable without a GPU, which is the reason the
allocator bookkeeping was kept backend-free in the first place.

Required cases:

1. **Borrow returns the arena address, not a copy.** Store known bytes, borrow,
   assert the returned pointer's contents match and that the pointer is stable
   across two successive borrow/release cycles for an untouched entry.
2. **Borrow misses** for an absent page, and for a resident page requested with
   the wrong size — both must return false and must not touch the out-pointer.
3. **A borrowed page is not evicted.** Fill the arena to capacity, borrow the
   LRU-front page, store a new page forcing an eviction, and assert the
   borrowed page's bytes are intact and that the *second*-oldest page was the
   victim instead.
4. **All-borrowed saturation fails the store cleanly.** Borrow every resident
   entry, attempt a store, assert it returns false and that no borrowed
   content changed.
5. **Deferred retirement.** Borrow a page, `erase()` it, assert `contains()` is
   immediately false and that the borrowed bytes are still readable; then
   `release()` and assert the slot is reused by the next same-size store.
6. **Re-store while borrowed does not alias.** Borrow page A, `store()` page A
   again with different bytes, assert the borrowed pointer still yields the
   *original* bytes (it must have been given a different slot).
7. **Refcount, not a flag.** Borrow the same page twice, release once, assert
   it is still protected from eviction; release again, assert it becomes
   evictable.
8. **Concurrency.** In the shape of the existing `test_host_tier_concurrency`,
   run borrow/release against concurrent store/erase and assert no borrowed
   region is mutated while held, and that used-bytes accounting returns to a
   consistent state at the end.

Also confirm the existing HostTier tests still pass unchanged — particularly
`test_host_tier_lru_eviction_order`, `test_host_tier_lookup_touch_keeps_mru`,
and `test_host_tier_lru_touch_is_not_linear_scan`, since eviction and LRU
handling are both being modified.

## 6. Build and verification

The library target is `llama-common`. **There is no target called `common`** —
asking for one exits 0 having done nothing and produces a false pass. The HIP
path must be compiled, not assumed: the zero-copy H2D lives inside
`#if defined(GGML_USE_HIP)`, and a CPU-only build will not touch it. Build
`build-hip` and report the real exit status; if the HIP build cannot be
completed, say so plainly rather than reporting the CPU build as verification.

Do not run any GPU workload, any inference, or any A/B measurement. No board
claim is held. Correctness only: unit tests plus a compiling HIP build. The
throughput re-measurement is a separate step, run later under a claim.

## 7. Shared-tree rules — non-negotiable

The tree has kmbandy's uncommitted work in it. Do **not** run
`git checkout`, `git restore`, `git stash`, `git reset`, `git add -A`, or
`git commit -a`. Stage only the files this task touches, by explicit path.
Leave `common/arg.cpp`, `common/common.cpp`, `tools/server/server-models.*`,
`docs/examples/router-fleet-*.ini`, `AGENTS.md`, `CLAUDE.md`, and everything
under the `dvgpr_occ` / memory-tier / paged-attn areas exactly as found. If a
file you need to touch already has uncommitted changes from someone else, stop
and report it instead of committing around it.

Expected files: `src/weight-pager/wp-host-tier.{h,cpp}`,
`src/weight-pager/wp-pager.{h,cpp}`, `tests/test-weight-pager.cpp`.
