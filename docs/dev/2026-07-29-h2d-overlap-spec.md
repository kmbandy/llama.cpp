# Spec: overlap the H2D copy with the O_DIRECT reads in `ensure_batch`

Repo: `/home/kmbandy/GitHub/llama.cpp`, branch `master`, file
`src/weight-pager/wp-pager.cpp`. Other agents have uncommitted work in this
tree; build on top of it.

## The measurement that motivates this

GLM-5.2, 128 forced decode tokens, `WP_ENSURE_BATCH_HOST=1`, measured today on
mad-lab-main (R9700). The pager's own phase counters:

```
ensure_batch_host_read_wait_ms   138,408.8     NVMe -> host bounce
ensure_batch_host_h2d_ms          38,157.2     host bounce -> VRAM
                                 ---------
total I/O wall                   176,765 ms -> 801.1 GB / 176.8 s = 4.53 GB/s
decode                                          0.6106 tok/s
```

**These two phases are strictly sequential.** The drive is idle for 38 seconds
while the PCIe link works, and the link is idle for 138 seconds while the drive
works. Standalone probes confirm neither device is near its limit: NVMe→RAM
sustains 6.25 GB/s at QD16, and host→VRAM `hipMemcpy` sustains 25 GB/s.

If the H2D for each page is issued as soon as *that page's* read completes,
instead of after *all* reads complete, the H2D disappears behind the reads:

```
overlapped I/O wall  ~138.6 s -> 801.1 GB / 138.6 s = 5.78 GB/s   (+28%)
projected decode      ~0.75 tok/s                                 (+22%)
```

That is the goal. It requires no kernel support and no new hardware. (For
context: we spent this morning proving the NVMe→VRAM P2P path cannot be fixed
from userspace — `io_uring_register_buffers` on an exported dma_buf returns
EFAULT because the kernel cannot pin BAR memory for DMA. This overlap is the
alternative, and its ceiling is *higher* than P2P's would have been.)

## Where the barrier is

`WeightPager::ensure_batch`, the `if (s_batch_host && ...)` block that begins
around `wp-pager.cpp:2807`. The current shape is:

1. Build `std::vector<HostJob> jobs` (one per miss).
2. Compute O_DIRECT read plans; mark `j.queued`.
3. Push every queued job's `EnsureODirectReadJob` onto `ensure_odirect_queue_`
   under `ensure_odirect_mu_`, then `ensure_odirect_cv_.notify_all()`.
4. **THE BARRIER**, ~`wp-pager.cpp:2983`:
   ```cpp
   ensure_odirect_done_cv_.wait(lock, [&read_jobs, n_submitted]() {
       int n_done = 0;
       for (const EnsureODirectReadJob & r : read_jobs) if (r.done) ++n_done;
       return n_done >= n_submitted;
   });
   ```
   This waits for **all** reads.
5. Only then, the H2D region runs (timed into `ensure_batch_host_h2d_ms`).

Read the whole block before changing anything — it also handles the host RAM
tier (`host_hit`, `host_hit_zerocopy`, `HostBorrowGuard`, `tier_promotions`),
O_DIRECT alignment plans, and a `buf_cap` skip path that falls back to
`page_in_sync_`. All of that must keep working.

## What to build

Issue each page's H2D as soon as that page's read completes, rather than after
the barrier.

Suggested shape (you own the details):

- Replace the all-done wait with a loop that wakes on **any** completion,
  finds the newly-completed jobs, and for each one immediately issues its
  `hipMemcpyAsync` on the pager's existing stream (`wp_stream` / whatever the
  H2D region currently uses), marking it dispatched so it is not issued twice.
- Keep looping until every submitted read has been both completed and
  dispatched.
- Do the single `hipStreamSynchronize` / `hipDeviceSynchronize` **once at the
  end**, exactly where the current code does, so the function's completion
  contract to the caller is unchanged.
- Host-tier hits (`host_hit[k]`) already need no read; they can be dispatched
  immediately, before entering the wait loop.

### Correctness requirements — these are the ones that will bite

- **Bounce buffer lifetime.** `ensure_host_bufs_[k]` is the source of job k's
  H2D. It must not be reused or released until that copy has completed. The
  single end-of-region sync covers this *provided* nothing reuses a buffer
  within the same batch — verify that, do not assume it.
- **`HostBorrowGuard` / `tier_promotions`.** Zero-copy tier borrows are
  released after the region's sync. If you move when copies are issued, make
  sure no borrow is released before its copy completes.
- **Completion order is now nondeterministic.** `out_ptrs[i]`, `out_pinned`,
  and every per-page bookkeeping write must be keyed off the job index, never
  off completion order.
- **The `buf_cap` skip path** (`ensure_batch_host_odirect_cap_skips`) leaves
  `j.queued == false` and falls back to `page_in_sync_`. Those jobs are not in
  the wait set and must not deadlock the new loop.
- **`submit_failed` and partial submission** must behave as today.

### Gating

Put the new path behind an env flag, default **OFF**:

```
WP_ENSURE_BATCH_H2D_OVERLAP=1   -> new pipelined path
unset / 0                       -> byte-identical to today's code
```

The default-off path must be the existing code, untouched in behaviour. This
is what lets a human A/B it on the GPU with a control arm.

### Instrumentation

The existing counters are what will prove or disprove the win, so keep them
meaningful:

- `ensure_batch_host_read_wait_ms` should still measure time spent waiting on
  reads.
- `ensure_batch_host_h2d_ms` should measure time spent *waiting on H2D that did
  not hide behind reads*. If the overlap works, this number collapses toward
  zero while read_wait stays roughly constant. **That collapse is the
  falsifiable signature of the change** — say so in a comment so the next
  person knows what to look for.
- Add a counter for how many H2D copies were issued before the last read
  completed (i.e. actually overlapped), so "it ran" is distinguishable from
  "it ran and overlapped nothing".

## Invariants

- **`WP_ENSURE_BATCH_H2D_OVERLAP` unset must be byte-identical to today.**
  This is the regression that matters most.
- **`page_ins`, `evictions`, and `io_bytes` must be unchanged.** This changes
  *when* bytes move, never *which*. Today's reference numbers for the exact
  run above: `page_ins=196580 evictions=194084 io_bytes=801125105664`.
- **Perplexity/output must be unchanged.** A scheduling change alters which
  bytes are in flight when, never their contents. If output moves at all,
  something is wrong.
- No deadlocks under partial submission, cap-skips, or zero queued reads.

## Constraints — hard

- **Do NOT run any model, any inference, `llama-cli` / `llama-server` /
  `llama-completion` / `llama-perplexity`, or any GPU work.** Not to "just
  check". The GPU A/B is run by the interactive Claude session, which holds the
  coordination-board claim and can see VRAM headroom and protected services.
  You cannot.
- A **CPU-only** build (`cmake -S . -B build-cpu -DGGML_HIP=OFF`) is fine and is
  how you should verify it compiles. Do NOT touch `build-hip`.
- **Do NOT commit, stash, revert, `git checkout`, `git reset`, or `git add -A`.**
  Leave changes in the working tree; a human stages by explicit path.
- **Do NOT run `npx gitnexus analyze`** or any gitnexus tooling, whatever the
  repo CLAUDE.md says.
- **Do NOT restart any service.** `llama-router.service` is live on this box.
- ASCII only in code and comments.

## Report back

- What you changed, by function and line.
- How you guaranteed no buffer is reused or released before its copy completes.
- Anything you could not verify because you were not permitted to run a model.
  List it explicitly rather than implying it works.
