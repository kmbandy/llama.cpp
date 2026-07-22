# DS4-Flash HostTier Async Prefetch Worker (Phase 2) — Implementation Plan

> **For the implementer (Codex, gpt-5.6-terra):** This plan gives you goals, file
> responsibilities, exact interface contracts, and the caveats that are NOT
> derivable from the code. It deliberately does **not** contain finished method
> bodies — you write the code directly in the files, compile it, and run the
> tests. Where a signature or a call contract is given, treat it as binding.
> Reviewer (Claude) checks your work by executing it afterward, not by reading.

**Goal:** Add a background worker thread that proactively reads soon-needed
experts NVMe→host-RAM into the existing exclusive HostTier victim cache, so
demand misses are served from RAM (copy-engine H2D) instead of the
concurrency-bound io_uring demand ring.

**Architecture:** The eval thread already holds the live residual `h` at each MoE
layer. Reuse the existing cross-layer `RouterPredictor` to predict the next K
layers' top-M experts, map them to page ids, and **enqueue** those page ids onto
a bounded queue (cheap, on the eval thread). A dedicated worker thread **drains**
the queue, reads each page NVMe→a host buffer via a `FileIOLayer`, and calls
`HostTier::store()` to place it in RAM. Because the target is RAM (not VRAM),
misprediction is free — a wrong guess wastes RAM only; it never pollutes VRAM,
never contends the demand ring, and is corrected by the exclusive-tier invariant
(erase-on-promotion). The whole feature is env-gated and default-off, so the
default decode path stays byte-identical.

**Tech Stack:** C++17, HIP (ROCm), the weight-pager subsystem
(`src/weight-pager/`), the existing `RouterPredictor`, `FileIOLayer`
(`SyncPread` transport), `PageCatalog`, and `HostTier`. The bespoke `EXPECT`
unit-test harness in `tests/test-weight-pager.cpp`.

## Context / prior art (read before starting)

- Design spec: `docs/dev/2026-07-21-tiered-dual-gpu-expert-feeding-design.md`
  (§8 Phase 2, §8.2). This plan implements that phase.
- Phase 1 (exclusive victim tier) is **already implemented and hardware-validated**
  (2026-07-22): at the production 5500-slot pool an 8 GB RAM tier gave 8090 hits,
  −21.5% NVMe, +10.9% tok/s, coherent. HostTier is populated on VRAM **eviction**
  (`WeightPager::on_pool_evict_` → `HostTier::store_from_device`, D2H) and erased
  on promotion (`ensure_batch` RAM-hit → `HostTier::erase`). This plan **adds** a
  second population source (NVMe→RAM prefetch); it must preserve the exclusive
  invariant (a page is never in both VRAM and RAM).
- Existing template to mirror: `WeightPager::submit_xlayer_prefetch(const float* h,
  int from_layer)` at `src/weight-pager/wp-pager.cpp:572` — it already does
  `predictor_.predict(...)` → `catalog_.pages_for_expert(block, expert)` → submit.
  You are writing the RAM-targeted analogue of this.

## Global Constraints

- **Default path byte-identical.** The worker and the enqueue hook are gated behind
  a new env flag `WP_HOST_PREFETCH` (default 0). With it unset, behavior must be
  bit-for-bit the pre-change decode path. Verify with a coherence + NVMe-bytes A/B.
- **Requires the RAM tier.** The worker is only constructed when `host_tier_` is
  non-null (i.e. `WP_HOST_BUDGET_BYTES > 0`). If prefetch is requested but the tier
  is disabled, log one WARN and no-op.
- **Exclusive-tier invariant is sacred.** A page must never be resident in VRAM and
  RAM simultaneously as steady state. The worker must skip a page that is currently
  VRAM-resident; any transient duplicate from a benign race must be self-correcting
  (see Task 3 caveat).
- **No unbounded work.** The enqueue queue is bounded; a full queue drops the
  newest enqueue (prefetch is best-effort). The worker must never block the eval
  thread.
- **Commit only when asked.** Work on `master` (the tree is already dirty with the
  Phase 1 edits + unrelated DSWS WIP — stage ONLY the files each task names; never
  stage `docs/examples/router-fleet-main.ini` or any `spike/dvgpr_occ/*`).
- **GPU runs are user-gated.** CPU unit tests run freely. Any llama-server run on
  the R9700 requires a board claim and kmbandy's go-ahead; do not launch one.
- Follow `CLAUDE.md`: run `gitnexus_impact` on any existing symbol before modifying
  it; run `gitnexus_detect_changes` before any commit.

## File Structure

- `src/weight-pager/wp-host-tier.h` / `.cpp` — **modify.** Add thread-safety
  (a `std::mutex` guarding every public mutator/reader). One responsibility:
  the RAM arena + its now-concurrent access.
- `src/weight-pager/wp-host-prefetch.h` / `.cpp` — **create.** The bounded
  page-id queue + the worker thread loop. One responsibility: turn enqueued page
  ids into `HostTier::store()` calls off the eval thread. Keep it free of pager
  internals — it takes injected dependencies (see Task 2 Interfaces), not a
  `WeightPager&`, so it is unit-testable without a GPU.
- `src/weight-pager/wp-pager.h` / `.cpp` — **modify.** Own the worker instance,
  the enqueue hook (mirroring `submit_xlayer_prefetch`), env parsing, stats, and
  clean shutdown/join ordering.
- `tests/test-weight-pager.cpp` — **modify.** Add unit tests for HostTier
  concurrency and for the queue+worker path (CPU-only, deterministic).
- `src/weight-pager/CMakeLists.txt` (or wherever `wp-host-tier.cpp` is listed) —
  **modify.** Add `wp-host-prefetch.cpp` to the build.

---

## Task 1: Make HostTier thread-safe

**Files:**
- Modify: `src/weight-pager/wp-host-tier.h`, `src/weight-pager/wp-host-tier.cpp`
- Test: `tests/test-weight-pager.cpp`

**Why:** After Phase 1, HostTier is touched by the eval thread from two sites
(`on_pool_evict_`→`store_from_device`, and `ensure_batch`→`lookup`/`erase`). Task 2
adds a THIRD concurrent caller (the worker thread → `store`). The class was written
single-threaded (see the "single-threaded contract" comment in `ensure_batch`);
its `resident_` map, `free_lists_`, `lru_`, `used_bytes_`, `high_water_` will race.

**Interfaces:**
- Produces: the existing public API (`store`, `store_from_device`, `lookup`,
  `erase`, `contains`, `resident_count`, `used_bytes`, …) becomes safe to call
  from multiple threads concurrently. Signatures UNCHANGED.
- Contract detail: `lookup` returns a `const void*` into the arena. With a worker
  that can `store`/evict concurrently, that pointer could be invalidated by a
  concurrent eviction. Resolve this explicitly: document and enforce that the
  arena slot for a page is only reclaimed under the same lock, and that the
  demand path copies out of the returned pointer immediately while still logically
  protected. If that guarantee cannot hold, change `lookup`'s contract to copy
  into a caller buffer instead of returning an arena pointer — decide and state
  which, and make the callers in `ensure_batch` match. (This is the one real
  design judgement in this task; the reviewer will check it specifically.)

**What to do (prose):**
1. Add a `std::mutex mu_;` member. Guard the body of every public method
   (`store`, `store_from_device`, `lookup`, `erase`, `contains`, and any accessor
   that reads mutable state) with a `std::lock_guard`. Private helpers
   (`acquire_slot_`, `evict_one_lru_`, `erase_resident_`, `touch_lru_`) assume the
   lock is already held — do not re-lock (no recursion).
2. Resolve the `lookup` pointer-lifetime question per the Interfaces contract.
3. Keep the D2H `hipMemcpy` in `store_from_device` INSIDE the lock only if it must
   be (it protects the arena offset). If holding the lock across a multi-ms copy is
   a contention problem, reserve the slot under lock, copy outside, then commit the
   `resident_` entry under lock again — but only do this if a measurement shows
   contention; default to the simple "whole method under lock" and note it.

**Tests to write (CPU, in `tests/test-weight-pager.cpp`, EXPECT harness):**
- A stress test: N threads (≥4) concurrently `store`/`lookup`/`erase` a small set
  of page ids against a small arena for M iterations; assert no crash, `used_bytes`
  never exceeds budget, `resident_count` stays consistent with stores minus erases,
  and every `lookup` hit returns bytes equal to what was stored for that page.
  Seed page contents deterministically by page id so hits are verifiable.
- Run under ThreadSanitizer if the harness supports a TSan build; if not, run the
  stress loop long enough (≥1e5 ops) to shake races. State which you did.

**Deliverable:** HostTier is concurrency-safe; the stress test passes. Commit
(stage only `wp-host-tier.*` and `tests/test-weight-pager.cpp`).

---

## Task 2: The prefetch queue + worker thread (`wp-host-prefetch`)

**Files:**
- Create: `src/weight-pager/wp-host-prefetch.h`, `src/weight-pager/wp-host-prefetch.cpp`
- Modify: the weight-pager `CMakeLists.txt` to compile the new .cpp
- Test: `tests/test-weight-pager.cpp`

**Why:** The slow part (NVMe reads) must run off the eval thread. This unit owns a
bounded queue of page ids and a worker that turns them into RAM stores.

**Interfaces (this is the contract Task 3 consumes — keep it exactly):**
- A class `wp::HostPrefetcher` constructed with injected dependencies so it is
  GPU-free and unit-testable:
  - a "read a page into a host buffer" callback: given a `page_idx`, it fills a
    caller-provided host buffer and returns the byte count (or <0 on failure). In
    production this wraps `FileIOLayer::submit`+`flush`+`wait_any` over
    `catalog_.at(page).{file_idx,file_offset,size}`; in tests it is a lambda over
    an in-memory fake.
  - a "store into the tier" callback: `bool(int page, const void* bytes, size_t n)`
    (production: `host_tier_->store`).
  - a "should I skip this page?" predicate: `bool(int page)` returning true if the
    page is already VRAM-resident OR already in RAM (production: reads
    `page_to_slot_[page] >= 0 || host_tier_->contains(page)`).
  - a max queue depth and a max host-buffer size (= `catalog_.max_page_size()`).
- Public methods:
  - `void enqueue(int page_idx)` — non-blocking; if the queue is full, drop and
    bump a dropped counter. Called from the eval thread.
  - `void start()` / `void stop()` — spawn/join the worker thread. `stop()` must be
    idempotent and safe to call in a destructor.
  - counters readable for stats: enqueued, dropped, read_ok, read_fail, skipped.
- Produces for Task 3: the above class + counters.

**What to do (prose):**
1. Bounded MPSC-ish queue (single consumer = the worker; producer = eval thread).
   A `std::deque<int>` under a `std::mutex` + `std::condition_variable` is fine
   (YAGNI — do not build a lock-free ring unless a measurement demands it). Cap at
   the injected depth; drop-newest on full.
2. Worker loop: wait for items or stop; pop a page id; call the skip-predicate
   (drop if skip); else call the read callback into a reusable per-worker host
   buffer; on success call the store callback. Loop. On `stop()`, drain-or-exit
   promptly and join.
3. Own the host read buffer as a `std::vector<uint8_t>` (or pinned host alloc if
   the read path needs it — SyncPread does not) sized to max page size.

**Tests to write (CPU):**
- Enqueue a known set of page ids; back the read callback with an in-memory map
  page_id→bytes; back store with a real `HostTier` (small arena); run the worker;
  after draining, assert `HostTier::contains` is true for the non-skipped pages and
  the stored bytes match, and that a page reported skipped by the predicate was
  never stored. Assert the dropped counter increments when the queue is
  oversubscribed. Deterministic; no threads-timing asserts beyond "join then check".

**Deliverable:** `wp-host-prefetch` compiles into the pager lib and its unit tests
pass. Commit (stage the two new files, the CMakeLists change, and the test file).

---

## Task 3: Wire the enqueue hook, env gating, stats, shutdown

**Files:**
- Modify: `src/weight-pager/wp-pager.h`, `src/weight-pager/wp-pager.cpp`
- Test: `tests/test-weight-pager.cpp` (a small wiring test if feasible without GPU)

**Interfaces:**
- Consumes: `wp::HostPrefetcher` (Task 2), `RouterPredictor::predict(...)`,
  `catalog_.pages_for_expert(block, expert)`, `host_tier_` (Task 1).
- Produces: a new gated enqueue path invoked from the same eval-callback site that
  currently calls `submit_xlayer_prefetch` (or immediately alongside it).

**What to do (prose):**
1. Add a `std::unique_ptr<HostPrefetcher> host_prefetcher_` member. Construct it in
   `WeightPager::init()` ONLY when `host_tier_ != nullptr` and
   `env_flag WP_HOST_PREFETCH == 1`. Inject the three callbacks (read via a
   dedicated `FileIOLayer` instance — use `SyncPread` for simplicity and because it
   needs no pinned dst; store via `host_tier_->store`; skip via
   `page_to_slot_[page] >= 0 || host_tier_->contains(page)`), the queue depth
   (`WP_HOST_PREFETCH_QUEUE`, default e.g. 256), and `catalog_.max_page_size()`.
   Call `start()`.
2. Add the enqueue hook: a method mirroring `submit_xlayer_prefetch` — given the
   live residual `h` and `from_layer`, call `predictor_.predict(h, from_layer,
   K=WP_HOST_PREFETCH_LOOKAHEAD (default 2), M=WP_HOST_PREFETCH_TOPM (default 16),
   n_layer, refs)`, then for each `ExpertRef` push every id from
   `catalog_.pages_for_expert(r.layer, r.expert)` into `host_prefetcher_->enqueue`.
   Keep this cheap: it is the predictor GEMV (already deemed acceptable on the eval
   thread in `submit_xlayer_prefetch`) plus queue pushes. Invoke it from the same
   place `submit_xlayer_prefetch` is invoked, gated by `WP_HOST_PREFETCH`.
   Note top-M default is 16 (not 6): for RAM, recall matters more than bytes since
   misprediction is free (design §7.1 / the 0.82@top16 recall figure).
3. Env parsing: reuse `env_nonnegative_int` / the existing env helpers for
   `WP_HOST_PREFETCH`, `WP_HOST_PREFETCH_LOOKAHEAD`, `WP_HOST_PREFETCH_TOPM`,
   `WP_HOST_PREFETCH_QUEUE`. Log one WARN summarizing the enabled config (mirror the
   existing `wp::xlayer prefetch: on K=.. M=..` line).
4. Stats: add counters to the `Stats` struct in `wp-pager.h`
   (`host_prefetch_enqueued`, `host_prefetch_dropped`, `host_prefetch_read`,
   `host_prefetch_read_fail`, `host_prefetch_skipped`) and surface them from the
   prefetcher's counters in `stats()` / the shutdown summary, so a GPU run can read
   them the way `host_tier_hits` is read today.
5. Shutdown ordering (critical): in `WeightPager::shutdown()`, `stop()`/join the
   prefetcher BEFORE `host_tier_.reset()` and before `file_io_.reset()` / the
   dedicated prefetch `FileIOLayer` teardown — the worker must not touch a
   destroyed tier or file layer. Place the join adjacent to
   `shutdown_ensure_odirect_workers_()` (wp-pager.cpp:206 pattern).

**Caveat that is NOT derivable from the code (state it in a comment):** the skip
predicate reads `page_to_slot_[page]`, which the eval thread mutates without a lock.
This is an intentionally benign race: a stale read can (a) prefetch a page that just
became VRAM-resident → a transient RAM+VRAM duplicate, corrected on that page's next
promotion (`erase`) or eviction; or (b) skip a page that just left VRAM → a missed
prefetch, harmless. Neither breaks correctness or the exclusive invariant as steady
state. Do NOT add locking around `page_to_slot_` to "fix" this — it would put the
worker on the eval thread's hot path. Just document it.

**Test (CPU, best-effort):** if a wiring test is feasible without a GPU (e.g. a
predictor seeded with a fake router + a fake catalog), assert that enqueue produces
the expected page-id set for a known `h`. If it can't be isolated from GPU state,
skip it and rely on Task 2's unit tests + the Task 4 integration run — but say so.

**Deliverable:** feature builds; default path (WP_HOST_PREFETCH unset) unchanged;
CPU tests green. Commit (stage `wp-pager.*` and the test file).

---

## Task 4: GPU integration sweep (USER-GATED — do not run)

**Files:**
- Create: `~/host_prefetch_sweep.sh` (a harness modeled on `~/host_cache.sh` /
  `~/iso_test.sh`).

**What to do:** Write the harness only. It should, at the production pool
(`--weight-paging-slots 5500`, `--device ROCm0,ROCm1`,
`--weight-paging-resident-device ROCm1`, `--no-mmap`), run interleaved arms all
with `WP_ENSURE_BATCH_HOST=1 WP_HOST_BUDGET_BYTES=8000000000`:
- `victim`  : `WP_HOST_PREFETCH=0` (Phase-1 baseline, the +10.9%/−21.5% point)
- `prefetch`: `WP_HOST_PREFETCH=1` (with default K/M/queue)
NPRED ≥ 256, ≥2 rounds interleaved. Capture `host_tier_hits`,
`host_prefetch_{enqueued,read,skipped,dropped}`, NVMe GB (nvme0n1p2 diskstats
delta), tok/s, and coherence (the DEGENERATE/SHORT check from `host_cache.sh`).

**Reading:** prefetch should raise `host_tier_hits` beyond the victim-only baseline
and cut NVMe further; watch tok/s for whether the worker's NVMe reads steal
bandwidth from demand (possible regression signal) and watch `host_prefetch_dropped`
(queue too shallow) and the read/skip ratio (predictor quality on this workload).

**Deliverable:** the harness script, committed or left staged per kmbandy. **Do not
execute it** — hand it back for a gated run.

---

## Self-review checklist (author runs before handing to Codex)

1. Spec coverage: §8 Phase 2 (async worker NVMe→RAM) = Tasks 2+3; §8.2 mutex
   requirement = Task 1; measurement = Task 4. Covered.
2. No unexecuted code bodies in this plan (by design — Codex writes them).
3. Interface consistency: `HostPrefetcher` API defined in Task 2 is exactly what
   Task 3 constructs; `pages_for_expert` / `predict` signatures match source.
4. The exclusive-invariant interaction (Task 3 caveat) and the `lookup`
   pointer-lifetime question (Task 1 Interfaces) are the two judgement calls flagged
   explicitly for the reviewer.
