# WP ensure_batch — multi-QD io_uring O_DIRECT read path (design + task)

**Date:** 2026-07-09
**Branch:** `feat/wp-dflash-ds4` (current WP working branch)
**Owner:** implementer = Codex; reviewer = Claude (+ GPT-5.6 Sol second pass)
**Related:** `docs/dev/2026-07-08-ds4flash-decode-levers.md`, `~/wp_logs/fable-consult-report.md`

## Goal
Cut DS4 Flash paged-decode expert-read time by making the weight-pager read experts via **multi-QD io_uring O_DIRECT**, instead of the current default **buffered** reads (O_DIRECT is stripped at `src/llama.cpp:279`, `wp::dup_clear_o_direct`). Measured on this exact box + model file:

| path (random 4.45 MiB pages, cold, compressed shard) | GB/s |
|---|---|
| io_uring **buffered** (≈ current WP default) | ~1.0–1.1 flat across QD1–24 |
| WP live `ensure_batch` (buffered + IOSQE_ASYNC kernel-worker farm) | ~3.0–3.65 |
| **io_uring O_DIRECT QD6 / QD16** | **5.7 / 5.9** |
| pread O_DIRECT QD6 / QD16 | 6.1 / 6.1 |
| seq O_DIRECT | 6.9 |

O_DIRECT hits ~6 GB/s **even on the zstd-compressed file** (near the seq ceiling), so **uncompressing the shards is NOT required** — the buffered path is the tax. Bench tool: `~/wp_logs/wp_io_bench <file>`.

Target: decode 1.67 t/s → ~2.3–2.6 t/s (I/O is ~72% of the ~555–600 ms token; ~1.5–1.65× read bandwidth on that fraction). This gain applies to **every** page_in.

## Root cause of the current O_DIRECT regression
An O_DIRECT host-bounce path already exists behind `WP_ENSURE_BATCH_HOST=1` (`src/weight-pager/wp-pager.cpp:1434-1590`) but it **regresses** (measured 1.37 t/s, `ensure_batch_gb_s`=1.58, `ensure_batch_submit_ms`=73190). Why: its read phase (`:1471-1516`) spawns **one pthread per miss** doing a **blocking `::pread`** (`:1494`), created in a sequential loop with a per-call join barrier. Sequential `pthread_create` + per-op barrier ⇒ low *effective* queue depth (nothing like the bench's true-simultaneous QD16 io_uring submit) ⇒ ~1.8 GB/s.

The P2P path already solved the equivalent problem for buffered reads with **one io_uring batch submit, `IOSQE_ASYNC | IOSQE_FIXED_FILE`** (`src/weight-pager/wp-file-io-p2p.cpp:123-135`, `submit_batch` `:242`). We need the same treatment for the O_DIRECT host-bounce read.

## The change
Replace the pthread-per-miss read phase in the `WP_ENSURE_BATCH_HOST` branch with a **single io_uring batch** of O_DIRECT reads:

1. **Dedicated O_DIRECT io_uring ring** for the host-bounce reads (the P2P ring reads into VRAM; this reads into host pinned bounce buffers, so it needs its own ring). Register the O_DIRECT fds (`ensure_odirect_fd_`, `:223`) with the ring (`IORING_REGISTER_FILES` → `IOSQE_FIXED_FILE`). Depth = `cfg_.io_uring_depth` (already 16).
2. For each miss: keep the existing **align-down** logic (`base = off & ~511`, `prefix`, `nbytes = pad512(prefix+size)`, `:1489-1499`) and the per-miss aligned pinned bounce buffer (`ensure_host_bufs_[k]`). Prep one `io_uring_prep_read(sqe, fixed_fd_idx, dst=bounce, nbytes, base)`, set `sqe->flags |= IOSQE_ASYNC | IOSQE_FIXED_FILE`.
3. **One `io_uring_submit()` for the whole batch**, then reap N CQEs (match by user_data → job index; verify `cqe->res == nbytes`). No per-miss thread.
4. Keep the existing **batched H2D** (`hipMemcpyAsync` per miss from `bounce+prefix` → VRAM slot, then one `hipDeviceSynchronize`, `:1519-1548`) and the mapping/stats updates unchanged.
5. Fallback: if the O_DIRECT open failed for a file (`ensure_odirect_fd_` returned -1) or a CQE errors, fall back to `page_in_sync_` for that miss (as the code already does at `:1544`).

Batches are small (a layer's ~18 misses), so one submit keeps ~18 O_DIRECT reads in flight — the bench regime that hit ~6 GB/s. The per-op barrier across layers remains (autoregression; out of scope here), but within-burst QD is what this fixes.

## Constraints (load-bearing)
- Branch `feat/wp-dflash-ds4`; commit per logical step; **do NOT push**. Verify `git branch --show-current`; if not on it, STOP.
- **No LLM inference** (GPU validation is human-gated). You MAY build and run `~/wp_logs/wp_io_bench`.
- HIP build: `cmake --build build-hip --target llama-server -j"$(nproc)"` (dual-arch gfx1201;gfx1030 already configured; no reconfigure unless new files). Must exit 0.
- Keep the new path behind `WP_ENSURE_BATCH_HOST=1` (do not change the default yet — that flip is a separate measured decision after A/B).
- Reuse liburing exactly as `wp-file-io-p2p.cpp` does (same include, same submit/wait idioms, same `IOSQE_ASYNC|IOSQE_FIXED_FILE`). Do not invent a new dependency.
- Preserve all existing stats semantics (`ensure_batch_submit_seconds` = O_DIRECT read phase, `ensure_batch_wait_seconds` = H2D phase).

## Validation gates (report each)
1. Build exits 0.
2. `wp_io_bench` unchanged (sanity of device numbers on the box).
3. Add a tiny standalone read-path microcheck if feasible (optional): a unit that issues the new io_uring O_DIRECT batch against one shard and prints GB/s — target ≥5 GB/s at batch≈16. If a standalone harness is impractical, say so and rely on the human GPU run.
4. **Report the exact human GPU-validation command** (do NOT run it): the `llama-server` hetero invocation with `WP_ENSURE_BATCH_HOST=1`, 6500 slots, and what to look for — `ensure_batch_submit_ms` should collapse from ~73000 to low hundreds, `ensure_batch_gb_s` should rise toward ~5, decode t/s vs the 1.67 baseline. Note: the previous serial run got 1.37 t/s / submit 73190 ms / 1.58 GB/s — those are the numbers to beat.

## Risks / notes
- **Compressed-extent fallback:** bench shows io_uring O_DIRECT ≈ 5.9 on the compressed file, so btrfs mostly honors it; if the in-engine number stalls near ~1.8, suspect per-inode-lock serialization on encoded extents and flag it (then uncompressing becomes a follow-up).
- **Bounce ring lifetime / registration:** register fds once (lazily, like `ensure_odirect_fd_`), not per call. Tear down in shutdown.
- **user_data matching:** tag each SQE with its job index; reap by CQE user_data, not submission order.
- Do not touch the P2P VRAM ring or the buffered default path.
