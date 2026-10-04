# P0 Instrumentation — Implementation Plan

> **For agentic workers:** implement one task at a time; each is dispatched to a
> fresh builder subagent and reviewed before the next begins.

> **No implementation code in this plan, by standing instruction** — code written
> into a plan gets rewritten by the builder anyway. Exact files, interfaces,
> behaviour and verification commands are specified instead.

**Goal:** make the pager's I/O instrumentation report what it actually measures,
per transport, so the transport A/B and the eventual throughput fix rest on
numbers that are true.

**Spec:** `docs/superpowers/specs/2026-07-24-p0-pager-io-throughput-design.md`

**Scope of this plan:** Unit 1 of the spec only. Unit 2 is a measurement task
needing a GPU claim; Unit 3 is deliberately unscoped until Unit 2 produces data.

---

## Global Constraints

Every task inherits these.

1. **Never run a model, inference, or any GPU workload.** No `llama-cli`,
   `llama-server`, `llama-completion`, `llama-perplexity`. Compiling is fine. If a
   step seems to need a GPU run, stop and report.
2. **Never restart a service; never `systemctl --user daemon-reload`.**
   `llama-router.service` is LIVE on mad-lab-main and must not be disturbed.
3. **This task changes NO behaviour.** Only what is measured and printed. If you
   find yourself altering read scheduling, worker counts, batching, fds, or
   eviction, stop — that is Unit 3, not this task.
4. **The working tree holds other people's uncommitted work**: `common/arg.cpp`,
   `tools/server/server-models.{cpp,h}`, `docs/examples/router-fleet-main.ini`, and
   files under `ggml/src/ggml-cuda/aiter-integration/.../dvgpr_occ/`. Never run
   `git checkout`, `git restore`, `git stash`, `git reset`, `git add -A`, or
   `git commit -a`. Stage only the files your task names. **A previous builder
   violated this and swept someone else's work into its commit — do not repeat it.**
5. **Never claim verification you did not perform.** Paste real command output. A
   previous builder claimed a compile check it never ran and shipped code that did
   not compile.
6. **Stop and report rather than improvise.** If the code contradicts this plan,
   halt and report what you found. A halted task with a clear report is a success.
7. Repo `~/GitHub/llama.cpp` on branch `master`. Do not push.

---

## Working environment

The repo is **not** on the machine the builder runs on. It is on **mad-lab-main,
100.86.191.92**, passwordless ssh, at `~/GitHub/llama.cpp`. The remote login shell
is **fish**, which breaks normal quoting — always wrap remote commands as
`ssh 100.86.191.92 'bash -lc "..."'`.

Local Edit/Write tools cannot touch remote files. For the large files in this task
(`wp-pager.cpp` is ~3900 lines), edit by writing a small Python patch script
locally that does targeted, unambiguous string replacement, `scp` it to `/tmp/` on
the remote, and run it there. Never rewrite a large file wholesale. After every
edit, inspect your own `git diff`.

**Build:** use the existing CPU build directory, which compiles the full library
and is far faster than HIP:
`ssh 100.86.191.92 'bash -lc "cd ~/GitHub/llama.cpp/build-cpu && cmake --build . --target llama-common -j 8"'`
The target is `llama-common` — there is no target named `common`, and asking for
one exits 0 having done nothing. Do **not** start a HIP build. Unit tests:
target `test-weight-pager`, then run `./bin/test-weight-pager` (no args, non-zero
on failure).

---

## Task 1 — Separate and correctly name the HOST-path phase timings

**Files:** `src/weight-pager/wp-pager.cpp`, `src/weight-pager/wp-pager.h`.

**Background the builder needs.** `ensure_batch` has three paths: a HOST O_DIRECT
pthread-pool path gated by `WP_ENSURE_BATCH_HOST`, a P2P `direct_to_device` path,
and a serial fallback. The HOST check runs first and returns unconditionally, so
when that env var is set the P2P path never executes. Both paths currently write
the **same** stats fields, `ensure_batch_submit_seconds` and
`ensure_batch_wait_seconds`, but they mean different things in each:

- On the P2P path they mean what they say — enqueue time and completion-wait time.
- On the HOST path, `submit` is computed to a point *after* the completion wait, so
  it is really the whole storage-read wall-clock; and `wait` is the leftover, which
  is the H2D copy.

**Required behaviour:**
- Introduce distinct stats fields for the HOST path covering, separately: building
  the job list, computing alignment/bounce parameters, enqueueing the jobs,
  blocking until all reads complete, and the H2D copy phase. Name them so each says
  what it measures.
- Leave the P2P path writing the existing `submit`/`wait` fields with their
  original meaning, and correct the `Stats` comment in `wp-pager.h` that currently
  labels the block "(P2P path)" while the HOST path also writes it.
- Print the new fields in the stats summary alongside the existing ones.
- **No behavioural change.** Do not alter thread counts, queue handling, batching,
  or read ordering. Timing instrumentation only.

**Steps:**
- [ ] Read `ensure_batch` end to end, both branches, plus the every-1000-calls
      phase log and the `Stats` struct. Confirm the mislabeling described above
      actually matches the code; if it does not, STOP AND REPORT rather than
      proceeding on my description.
- [ ] Add the new fields and populate them on the HOST path.
- [ ] Correct the misleading comment; print the new fields.
- [ ] Build `llama-common`; paste real output showing `wp-pager.cpp.o` rebuilt.
- [ ] Run `test-weight-pager`; paste the tail.
- [ ] Commit only the two named files, with trailers:
      `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
      `Claude-Session: https://claude.ai/code/session_01JtyuKaPrB6ZXvZroDxoeK4`

---

## Task 2 — Transport identity and a comparable headline metric

**Files:** `src/weight-pager/wp-pager.cpp`, `src/weight-pager/wp-pager.h`.

**Required behaviour:**
- Record which transport actually served reads — HOST pthread pool, P2P direct,
  or serial fallback — and emit it in the teardown summary as a single explicit
  line. Nothing in today's logs states this plainly, which is exactly how the
  confusion about which path was running survived this long.
- Provide one headline metric comparable across transports: bytes genuinely read
  from storage divided by wall-clock spent in the storage-read phase (excluding
  H2D and excluding HostTier hits, which read no storage bytes). If an equivalent
  field already exists, verify it is computed consistently on both paths and fix it
  if not, rather than adding a duplicate.
- Fix the known contamination: `ensure_batch_n_sub_sum` must count only real
  submissions and must not fold in HostTier hits; `ensure_batch_max_n` must not be
  derived from a total that includes host hits. Report host hits separately.

**Steps:**
- [ ] Read how `io_effective_gb_s`, `ensure_batch_gb_s`, `ensure_batch_n_sub_sum`
      and `ensure_batch_max_n` are currently computed on each path.
- [ ] Implement transport identity + the corrected headline metric + the
      contamination fix.
- [ ] Build and run the tests as in Task 1; paste real output.
- [ ] Commit only the two named files with the same trailers.

---

## Task 3 — Achieved-concurrency counter, with unit tests

**Files:** `src/weight-pager/wp-pager.cpp`, `src/weight-pager/wp-pager.h`,
`tests/test-weight-pager.cpp`.

**Why this matters.** `avg_n` and `max_n` count how many jobs were *queued* per
batch, not how many reads were genuinely *in flight*. The distinction is the whole
question: the pool has 16 workers and averages ~9 queued jobs per batch, yet
delivers well under what the drive does at that depth. Without an in-flight
measure there is no way to tell whether the workers are actually running
concurrently.

**Required behaviour:**
- Track reads genuinely in flight on the HOST path — incremented when a worker
  begins its `pread` and decremented when it completes — and report both the peak
  and a time- or sample-weighted average. Use a cheap atomic; this sits on the hot
  path and must not become a bottleneck itself.
- Report it in the teardown summary next to the queue-occupancy figures, labelled
  so the two cannot be confused.

**Testing.** `tests/test-weight-pager.cpp` uses a hand-rolled harness: each test is
a `static int test_*()` returning a failure count, using the file's existing
`EXPECT` / `EXPECT_EQ_INT` macros, registered in a table inside `main()`. Follow
those conventions exactly.

- [ ] Write failing tests first, for whatever pure accounting logic this
      introduces — peak and average in-flight computed from a synthetic sequence of
      begin/end events, and confirmation that HostTier hits contribute zero storage
      bytes and zero submissions to the headline metric. If the accounting cannot
      be reached without a live worker pool, extract the arithmetic into a small
      pure helper so it can be tested, and say so in your report.
- [ ] Build and run; confirm the new tests FAIL; paste the output.
- [ ] Implement.
- [ ] Build and run; confirm ALL tests pass including pre-existing ones; paste it.
- [ ] Commit only the three named files with the same trailers.

---

## After this plan

Unit 2 of the spec — the HOST vs P2P transport A/B with the RAM tier on and off,
arms interleaved and their order alternated between rounds — is a measurement task
requiring a GPU claim and kmbandy's go-ahead. Unit 3, the actual throughput fix, is
scoped only once that data exists.
