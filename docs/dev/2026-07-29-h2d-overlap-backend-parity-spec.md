# Spec: make the H2D/read overlap backend-agnostic (Vulkan parity)

Repo: `/home/kmbandy/GitHub/llama.cpp`, branch `master`. The overlap landed in
commit `63dfe38d0`; this fixes its backend coverage.

## The problem

`WP_ENSURE_BATCH_H2D_OVERLAP=1` (wp-pager.cpp, `ensure_batch` host path) issues
each page's host->VRAM copy as that page's read completes. Measured on GLM-5.2:
decode 0.6083 -> 0.7049 t/s, `h2d_ms` 38,188 -> 7,986, with 95% of copies going
out before the last read landed.

It is gated to HIP/CUDA only:

```cpp
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
#if defined(GGML_USE_VULKAN)
    const bool batch_h2d_overlap = s_batch_h2d_overlap != 0 && !transport_.is_vulkan();
#else
    const bool batch_h2d_overlap = s_batch_h2d_overlap != 0;
#endif
#else
    const bool batch_h2d_overlap = false;
#endif
```

The exclusion exists because the overlap path calls `hipMemcpyAsync` DIRECTLY
rather than going through `GpuTransport`. That is the actual defect: a pager
feature reached around its own backend abstraction, and Vulkan fell out.

This fleet has four GPUs across two machines: R9700 (HIP), RX 6900 XT (HIP),
GTX 1070 (CUDA), RX 480 (Vulkan). One of four is currently excluded, and the
cross-machine plan needs all of them. Backend parity ships WITH a feature, not
after it.

Note this is the FIFTH instance of a recorded bug class in this subsystem --
`#if defined(GGML_USE_*)` used where a RUNTIME backend check was needed (prior
four: WP_ENSURE_BATCH_HOST, the routing-index hipMemcpy, HostTier::
store_from_device, and the Vulkan expert-offset publication fixed in
3b90dd346). Treat it as a missing invariant, not a one-off.

## What already exists (do not rebuild it)

`GpuTransport` (src/weight-pager/wp-gpu-transport.{h,cpp}) already provides
async staging with per-copy completion on ALL THREE backends:

- `int stage_in_async(void * dst, const void * src_pinned, size_t payload_size,
  size_t slot_size)` -- issues the copy, returns an event handle, does NOT
  wait. The Vulkan branch calls `ggml_backend_vk_wp_stage_in()` and returns a
  fence; its comment explicitly records that an earlier version waited inline
  and that this "collapsed the batch paths to one page in flight at a time".
- `bool synchronize(int evt)` -- wait for one event.
- `bool query(int evt)` -- poll one event.
- `void release_event(int evt)` -- return the handle to the pool.
- `bool is_vulkan()` -- runtime backend check.

So the Vulkan capability is present and tested. What is missing is that the new
overlap path does not use it.

## What to build

Route the overlap's per-page copy through `GpuTransport::stage_in_async()`
instead of calling `hipMemcpyAsync` directly, then wait on the collected events
once at the end of the region instead of the single device-wide sync.

- Keep an event handle per job index (nondeterministic completion order is
  already handled; keep it that way).
- At the end of the region, wait for every outstanding event, then release
  every handle. Release must happen on every exit path, including failures --
  a leaked handle exhausts the pool and `stage_in_async` starts returning -1,
  which would silently degrade the batch.
- Delete the `is_vulkan()` exclusion and the surrounding `#if defined(...)`
  ladder. The gate becomes: overlap enabled if the env is set and the transport
  supports async staging, decided at RUNTIME.

### The event pool is the thing most likely to break this

`stage_in_async` fails when `free_events_` is empty, logging "event pool
exhausted (queue depth too small?)". The measured batch width on GLM is
`ensure_batch_avg_n = 19.72`, and the max is larger. The pool is sized by the
`n_events` argument to `GpuTransport::init`. Check what it is actually set to
on each backend and make sure a full batch's worth of copies can be in flight;
if not, size it from the batch width rather than a constant. If the pool can
still be exhausted, the code must fall back cleanly to the synchronous path for
the remaining pages of that batch -- never drop a copy, never return a NULL
pointer for an active expert.

### Other requirements

- **Default OFF is unchanged and must stay byte-identical to the pre-63dfe38d0
  barrier path.** This is the regression that matters most and it is what makes
  the A/B control arm valid.
- The existing counters must keep working on every backend:
  `ensure_batch_host_read_wait_ms`, `ensure_batch_host_h2d_ms`, and
  `copies_before_last_read`. The last one is what distinguishes "ran" from
  "ran and overlapped nothing" -- on Vulkan it is the ONLY way we will know the
  fences are actually being issued early rather than serialized.
- No behaviour change for HIP/CUDA beyond the indirection. The HIP A/B numbers
  above must reproduce; if they move, the abstraction is costing something and
  that needs reporting, not absorbing.

## Invariants

- `WP_ENSURE_BATCH_H2D_OVERLAP` unset -> byte-identical to today.
- `page_ins`, `evictions`, `io_bytes` unchanged. Reference for the GLM run:
  `page_ins=196580 evictions=194084 io_bytes=801125105664`. This changes WHEN
  bytes move, never WHICH.
- No event-handle leak on any path, including read failures, cap-skips,
  partial submission, and zero queued reads.
- No deadlock when the event pool is exhausted mid-batch.

## Constraints -- hard

- **Do NOT run any model, any inference, `llama-cli` / `llama-server` /
  `llama-completion` / `llama-perplexity`, or ANY GPU work.** The GPU A/B is
  run by the interactive Claude session, which holds the coordination-board
  claim and can see VRAM headroom, live claims and protected services. You
  cannot. Standing fleet rule, not negotiable.
- CPU-only build to verify compilation:
  `cmake -S . -B build-cpu -DGGML_HIP=OFF && cmake --build build-cpu -j 16`.
  Do NOT touch `build-hip` or `build-vk`.
- **Do NOT commit, stash, revert, `git checkout`, `git reset`, `git add -A`.**
  Leave changes in the working tree; a human stages by explicit path.
- **Do NOT run `npx gitnexus analyze`** or any gitnexus tooling.
- **Do NOT restart any service.** `llama-router.service` is live.
- ASCII only in code and comments.

## Report back

- What you changed, by function.
- What `n_events` resolves to per backend, and what happens when a batch is
  wider than the pool.
- Where event handles are released, and your argument that none can leak.
- Everything you could not verify because you could not run a model. List it
  explicitly; do not describe untested code as working.
