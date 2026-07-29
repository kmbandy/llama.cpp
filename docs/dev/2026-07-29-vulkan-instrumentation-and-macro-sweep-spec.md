# Spec: make the weight pager's Vulkan path measurable, and sweep the macro-vs-runtime backend checks

Repo: `/home/kmbandy/GitHub/llama.cpp`, branch `master`, at `dfad47e27` (both
fleet machines and origin are converged on this commit).

This is two jobs in one because the second causes the first. Do them in order.

---

## PART 1 (the blocker): the Vulkan paging path reports nothing

### The measurement

RX 480 (`Vulkan0`, RADV POLARIS10), LFM2.5-8B-A1B-UD-Q5_K_S, 256 slots, 128
decode tokens, `build-army` (CUDA+Vulkan), bracketed ctl/overlap/ctl2. From the
run's own summary:

```
host_batches            = 2901        <- the HOST block IS entered
inflight_peak           = 8
inflight_avg_at_read_start = 2.62     <- reads ARE genuinely concurrent
page_ins                = 35568
io_bytes                = 90006110208 <- 90 GB really was read

ensure_batch_calls      = 0           <- ALL of these are zero
ensure_batch_pages      = 0
ensure_batch_avg_n      = 0.00
ensure_batch_gb_s       = 0.000
ensure_batch_host_jobs_ms    = 0.0
ensure_batch_host_prep_ms    = 0.0
ensure_batch_host_enqueue_ms = 0.0
ensure_batch_host_read_wait_ms = 0.0
ensure_batch_host_h2d_ms       = 0.0
copies_before_last_read = 0           <- with 2901 overlap batches
```

Zero-millisecond reads of 90 GB are impossible, so these counters are not
measuring a fast path -- they are not being reached at all. On HIP, the same
commit and the same code report `ensure_batch_calls = 9968` and
`copies_before_last_read = 186556 / 196576`.

### What to find out

`stats_.ensure_batch_host_path_batches` is incremented near the top of the HOST
block and DOES show 2901. The three `++stats_.ensure_batch_calls` sites are at
`wp-pager.cpp:3654`, `:3732`, `:3866` and are never reached on Vulkan. So
control leaves the HOST block, or bypasses those sites, somewhere in between.

A likely culprit is already commented in that block: its non-HIP/CUDA `#else`
path "ignores `jobs`/`host_hit` entirely in favor of `page_in_sync_`". Note that
in a `build-army` build `GGML_USE_CUDA` IS defined, so the compile-time branch
taken is the CUDA one even though the device at runtime is Vulkan -- exactly the
mismatch described in Part 2. Verify this rather than assume it; the point of
this task is to establish the real path, not to confirm my guess.

### What to build

Make the HOST-path instrumentation correct on every backend:

- `ensure_batch_calls`, `ensure_batch_pages`, `avg_n`, `gb_s`, and the
  `host_*_ms` phase timers must reflect real work on Vulkan exactly as they do
  on HIP.
- `copies_before_last_read` must count overlapped copies on Vulkan, or -- if the
  overlap genuinely cannot happen there -- the code must say so explicitly in a
  log line rather than silently reporting 0 alongside a non-zero batch count.
  **Silence that reads as "it ran fine" is the failure mode we are removing.**
- If the Vulkan device really does take a `page_in_sync_` per-page path instead
  of the batch path, then `host_batches` must NOT be incremented for it. A
  counter that says "the batch path ran" while the batch path did not run is
  worse than no counter.

Do not "fix" this by making the numbers non-zero. Establish which path executes,
then instrument that path honestly.

---

## PART 2 (the cause): sweep compile-time backend checks that should be runtime

`src/weight-pager/` contains **55** occurrences of `defined(GGML_USE_*)`:

```
wp-gpu-transport.cpp 15   wp-eval-cb.cpp 16   wp-pager.cpp 10
wp-host-tier.cpp      4   wp-prefetch.cpp  3   wp-file-io-p2p.cpp 2
wp-pool.cpp           2   wp-gpu-runtime.h 2   wp-pager.h 1
```

A `GGML_USE_*` macro says what was COMPILED, never what is RUNNING. `build-army`
compiles CUDA + Vulkan + RPC together, so on that build `GGML_USE_CUDA` and
`GGML_USE_VULKAN` are both defined while the actual device may be either.

This has now produced **six** separate live bugs in this one directory:

1. `WP_ENSURE_BATCH_HOST` short-circuiting ahead of the P2P check
2. the routing-index `hipMemcpy` under a satisfied-but-wrong guard
3. `HostTier::store_from_device` copying from a Vulkan sentinel pointer
4. the Vulkan expert-offset publication gated on `#if defined(GGML_USE_VULKAN)`
   alone -- logged ~100M lines / 200 MB in two minutes on a CUDA-only run
   (fixed in `3b90dd346` by switching to `pager->is_vulkan()`)
5. the h2d-overlap Vulkan exclusion (fixed in `159368422`)
6. this instrumentation gap

A sweep was recommended on 2026-07-27 and never done. Do it now.

### What to build

Go through all 55 sites and classify each one:

- **Legitimately compile-time** -- guarding an `#include`, a type that only
  exists under that backend, or a call to an API absent from the build. Leave
  it, and add a one-line comment saying why it is compile-time, so the next
  reader does not have to re-derive it.
- **Should be runtime** -- the code is asking "which device am I talking to?"
  Convert it to the existing runtime accessors: `GpuTransport::is_vulkan()`,
  `is_initialized()`, or a new accessor if one is genuinely needed. Precedent to
  follow: `3b90dd346`.

Report the classification as a list. I want to see the count of each and the
reasoning for anything non-obvious. If a site is ambiguous, say so rather than
guessing -- an ambiguous guard that gets silently "fixed" wrong is how number
seven happens.

---

## Invariants

- **HIP behaviour must not change.** The measured reference on GLM-5.2, 128
  decode tokens, `WP_ENSURE_BATCH_HOST=1`: overlap OFF 0.6054 t/s, overlap ON
  0.7189 t/s, `h2d_ms` 39,180 -> 2,580, `copies_before_last_read` 186,556 of
  196,576, `page_ins` 196,580, `io_bytes` 801,125,105,664. I will re-run this;
  if it moves, the sweep broke something.
- `WP_ENSURE_BATCH_H2D_OVERLAP` unset must stay byte-identical to the barrier
  path.
- Counters may only change where they were previously WRONG. A counter that was
  right on HIP must stay right on HIP.

## Constraints -- hard

- **Do NOT run any model, any inference, `llama-cli` / `llama-server` /
  `llama-completion` / `llama-perplexity`, or ANY GPU work.** Both GPU A/Bs are
  run by the interactive Claude session, which holds the coordination-board
  claims and can see VRAM headroom, live claims and protected services. You
  cannot.
- CPU-only build to verify compilation:
  `cmake -S . -B build-cpu -DGGML_HIP=OFF && cmake --build build-cpu -j 16`.
  Do NOT touch `build-hip`, `build-vk`, or `build-army` -- those are the GPU
  builds the fleet's live binaries come from.
- **Do NOT commit, stash, revert, `git checkout`, `git reset`, `git add -A`.**
  Both machines and origin are converged at `dfad47e27`; a stray commit or
  branch move desynchronises them. Leave changes in the working tree.
- **Do NOT run `npx gitnexus analyze`** or any gitnexus tooling.
- **Do NOT restart any service.**
- ASCII only in code and comments.

## Report back

- Part 1: which code path the Vulkan device actually takes through the HOST
  block, and what you changed so it reports honestly.
- Part 2: the 55-site classification -- how many compile-time-correct, how many
  converted to runtime, how many ambiguous and why.
- Everything you could not verify because you could not run a model. List it
  explicitly. Do not describe untested code as working.
