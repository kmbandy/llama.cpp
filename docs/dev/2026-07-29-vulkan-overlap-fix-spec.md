# Spec: make the H2D overlap actually work on Vulkan

Repo: `/home/kmbandy/GitHub/llama.cpp`, `master` at `11fb5c38b` (both machines
and origin converged).

Fix the feature. This is not an instrumentation task -- the instrumentation from
the previous handoff works and is what localised the problem.

## What the instrumentation now says

RX 480 (`Vulkan0`, RADV POLARIS10), LFM2.5-8B-A1B, 256 slots, 128 decode tokens,
warm-up pass discarded, bracketed ctl/overlap/ctl2:

```
HOST runtime backend=vulkan  h2d_route=transport-overlap
HOST H2D OVERLAP: NOT OBSERVED - batches=2901 copies_before_last_read=0
stage_submissions=0  stage_completions=0  sync_fallback_pages=35568
page_ins=35568                                   <- ALL pages fell back
decode ctl 3.2878 / overlap 3.3057 / ctl2 3.3024 <- no effect, correctly
```

## There are TWO problems. Do them in order.

### Problem 1 -- `stage_submissions = 0`, and that is a bug

`issue_overlap_copy` (`wp-pager.cpp:3063`) increments `batch_h2d_submissions`
only after `transport_.stage_in_async()` returns >= 0. Zero submissions with all
pages falling back means it never succeeded once.

**Do not assume this is the unpinned-source path.** I read the return path:
`ggml_backend_vk_wp_stage_in` (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:8363`)
returns **true** for an unpinned source -- it creates the fence, submits, calls
`ggml_backend_vk_wp_event_wait()` internally, and hands back an already-signalled
event (`ggml-vulkan.cpp:8432-8435`, then `return true` at the end). Forced
synchronous, but successful. So submissions SHOULD have incremented.

The failure is therefore upstream. Candidates, all of which produce the identical
counter signature today:

1. `stage_in_async`'s own guards: `!initialized_`, `payload_size > slot_size`, or
   `free_events_.empty()` ("event pool exhausted").
2. Inside `ggml_backend_vk_wp_stage_in`: `buffer_ctx == nullptr ||
   buffer_ctx->dev_buffer == nullptr`, or the bounds check
   `offset = (uintptr_t) dst - (uintptr_t) vk_ptr_base;
    if (offset > dst_buffer->size || slot_size > dst_buffer->size - offset)`.
   That offset arithmetic depends on a global `vk_ptr_base`; if `dst` is not
   within it the offset is garbage and this returns false.
3. `ggml_vk_buffer_write_async` returning false.
4. `overlap_source(k)` returning `nullptr`, so `stage_in_async` is never called
   at all. **Note this exit does NOT close `overlap_async_submission_open`,**
   whereas a `stage_in_async` failure does -- yet both yield
   `submissions=0, fallback=all`. The counters cannot currently tell these apart.

**First deliverable: make them distinguishable.** A one-shot (rate-limited)
warning at each of those exits naming which one fired, on the first occurrence
per run. Then the cause is a single run away instead of a guess. Add the same to
`stage_in_async`'s three guards -- today two of them return -1 silently.

Then fix whatever it turns out to be.

### Problem 2 -- even when it submits, Vulkan will not overlap

The Vulkan implementation documents this itself, at `ggml-vulkan.cpp:8388`:

> Can this transfer safely be left in flight? Only if `src` is host memory
> registered with THIS Vulkan device, in which case the copy reads straight from
> it. Otherwise `ggml_vk_buffer_write_async` routes through the single
> device-wide `device->sync_staging` buffer, and two overlapping transfers would
> each memcpy into that same region -- the second clobbering data the first's
> queued copy has not executed yet. So an unpinned source MUST be fenced before
> this function returns...
>
> **The pager's O_DIRECT bounce arena is currently hipHostMalloc'd, which this
> device knows nothing about, so today it always takes the fenced path.
> Register that arena with Vulkan and this becomes genuinely async with no other
> change.**

`ggml_vk_host_get()` cannot find `ensure_host_bufs_` (the O_DIRECT bounce arena
the overlap path reads from), so `src_is_pinned` is false and every transfer is
fenced inline. The forced fence is a CORRECTNESS constraint, not an oversight --
do not remove it.

The previous handoff registered host-tier and prefetch staging with the
transport, but **not** the O_DIRECT bounce arena, which is the source the overlap
path actually uses.

**Second deliverable: register the O_DIRECT bounce arena with the Vulkan device**
so `ggml_vk_host_get` finds it and `src_is_pinned` becomes true. Find the
existing host-register entry point (whatever `ggml_vk_host_get` looks up against
-- there will be a matching register/unregister pair) and call it when the arena
is allocated, unregistering before teardown. Match the lifetime handling already
used for the transport-owned allocations added last time.

If registration is not possible for this arena, say so and explain why rather
than working around it -- an unpinned source that is NOT fenced is silent data
corruption, which is far worse than no overlap.

## Invariants

- **HIP must not regress.** GLM-5.2, 128 decode tokens, `WP_ENSURE_BATCH_HOST=1`:
  overlap OFF 0.6054 t/s, ON 0.7189, `h2d_ms` 39,180 -> 2,580,
  `copies_before_last_read` 186,556/196,576, `page_ins` 196,580, `io_bytes`
  801,125,105,664.
- `WP_ENSURE_BATCH_H2D_OVERLAP` unset stays byte-identical to the barrier path.
- **Never leave an unpinned Vulkan transfer in flight.** If the source is not
  registered with the device, it MUST be fenced before return.
- The Vulkan fallback must stay correct. Falling back is acceptable; corrupting
  a slot is not.

## Constraints -- hard

- **Do NOT run any model, any inference, `llama-cli` / `llama-server` /
  `llama-completion` / `llama-perplexity`, or ANY GPU work.** Both A/Bs are run
  by the interactive Claude session, which holds the board claims and can see
  VRAM headroom, live claims and protected services. You cannot.
- CPU-only build to verify compilation:
  `cmake -S . -B build-cpu -DGGML_HIP=OFF && cmake --build build-cpu -j 16`.
  Do NOT touch `build-hip`, `build-vk`, or `build-army`.
- **Do NOT commit, stash, revert, `git checkout`, `git reset`, `git add -A`.**
  Both machines and origin are at `11fb5c38b`; a stray commit desynchronises
  them. Leave changes in the working tree.
- Uncommitted DSWS kernel-spike work under
  `ggml/src/ggml-cuda/aiter-integration/` is NOT yours -- do not touch it.
- **Do NOT run `npx gitnexus analyze`.**
- **Do NOT restart any service.** `llama-router.service` is live on both boxes.
- ASCII only in code and comments.

## Report back

- Which of the four candidates in Problem 1 actually fired, and how you
  established it.
- Whether the O_DIRECT bounce arena can be registered with the Vulkan device,
  and where you register/unregister it.
- Everything you could not verify without a GPU. Do not describe untested code
  as working.
