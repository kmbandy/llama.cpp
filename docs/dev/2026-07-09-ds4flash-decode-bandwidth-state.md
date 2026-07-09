# DS4 Flash Paged Decode — Bandwidth Investigation State + Next Levers

**Date:** 2026-07-09
**Branch:** `feat/wp-dflash-ds4`
**Prior docs:** `2026-07-08-ds4flash-decode-levers.md` (Grok's lever catalog), `2026-07-08-wp-hetero-dflash-oracle-plan.md`
**KG anchors (mneme):** `1f71eae7` (ground-truth bandwidth), `c06aa208` (eviction analysis), `ee5cdbda`/`a0b5cc1d` (O_DIRECT bench), `5e8f5676` (DFlash arc)

## TL;DR
Current best single-stream decode = **~1.84 t/s** (buffered P2P path), up from 0.004 t/s historically (~460×). We are **I/O-bound**: ~66-87% of each token is expert paging. This session exhaustively tested the I/O-bandwidth and eviction levers; the headline is a **measured landscape**, NOT a declared wall. The one unsolved, high-value anomaly is a **2.79× O_DIRECT read amplification in-engine** — cracking it is the most likely path past 1.84.

## The workload (measured)
- DS4 Flash UD-Q8_K_XL, ~162 GB, 284B/13B-active, 256 routed experts (top-6+shared), 43 layers, hyper-connections.
- Hetero placement: attention/FA/KV/lm_head/draft on 6900XT (eGPU/TB3), routed experts paged NVMe->R9700 (PCIe4 x16), token_embd on CPU. `--parallel 1`.
- Expert page = 4.45 MiB (one gate/up/down sister). Pool = 6500 slots (~27.6 GiB); 7000 OOMs.
- Per 128-token decode: 28,727 page_ins (~224/tok), ~128 GB payload. Compute floor ~180 ms/tok. Hardware: ONE WD_BLACK SN850X (PCIe4 x4, ~7 GB/s), one SATA HDD (useless), 15 GB RAM (~11 free).

## Scoreboard (all measured this session, 128-tok Roman-Empire decode, 6500 slots)
| config | t/s | page_ins | eff GB/s | notes |
|---|---|---|---|---|
| **buffered P2P (default, BEST)** | **1.84** | 28,727 | ~2.2 physical, 1.13x amplification | the winner |
| smart eviction (SLRU/2Q/LFU/structural) | — | — | — | DEAD: sim shows all <= LRU; Belady gap is content-timed, unpredictable |
| RAM victim tier (HostTier 5GB) | 1.63 | 28,727 | — | NO-OP: stores on READ (redundant w/ VRAM), hits=0; disables P2P |
| O_DIRECT io_uring (compressed) | 1.10-1.21 | 28,727 | 1.4 | submit stall 52s + 2.79x amplification |
| O_DIRECT worker-pool (Codex fix) | 1.33 | 28,727 | 1.55 payload | submit stall FIXED; 2.79x amplification remains |
| O_DIRECT worker-pool (UNCOMPRESSED shards) | 1.39 | 28,727 | 1.55 | **2.79x amplification UNCHANGED -> compression NOT the cause** |
| spec-decode DFlash (buffered, VERIFY_MAX=0, n_max=4) | 1.20 | 62,538 | 3.42 | **76% accept (mean 3.97/4)** but 2.18x byte cost |
| spec-decode + O_DIRECT | 0.62 | 62,538 | 1.47 | both failure modes stack |

## Key findings (with evidence)
1. **Eviction is dead.** Instrumented routing trace (`WP_ROUTE_TRACE` env, still in tree) + offline sim (`~/wp_logs/analyze_v2.py`). LRU@6500=27903 (validated vs live 28727); Belady@6500=19306 (-31%) but **no realizable policy captures it** (SLRU/2Q worse, LFU -2.3%, structural worse). Misses are content-timed. Also: **misses are ~100% novel** (0.0% in prev-token same layer, 0.1% in last-8-tokens) -> no cheap prefetch/overlap signal. No cache convergence at long context (misses/tok plateau ~190).
2. **The WP `ensure_batch_gb_s` metric is MISLEADING.** Device ground truth via `/proc/diskstats` (field 6 sectors*512): device reads 3.0-3.7 GB/s in-engine while WP reported 1.35. Do NOT trust ensure_batch_gb_s; measure diskstats.
3. **O_DIRECT submit stall — FIXED by Codex.** `io_uring_submit` was blocking ~52s (synchronous O_DIRECT submission in the HIP-active process). Codex replaced it with a persistent blocking-`pread` worker pool (behind `WP_ENSURE_BATCH_HOST=1`, `WP_ODIRECT_READ_WORKERS`). submit 52s -> ~0.1s. Report: `~/wp_logs/codex-iofix-report.md`.
4. **O_DIRECT amplifies 2.79× in-engine — UNSOLVED (the key open anomaly).** Physical reads = 356 GB for 128 GB payload = 12.4 MB physical per 4.45 MB page. Consistent across io_uring AND worker-pool, compressed AND uncompressed. Standalone microbench (`/tmp/odirect_*.cpp`, hipcc+liburing) reads 1× at 5.9 GB/s. **Buffered path = 1.13× (efficient).** NOT compression (uncompressed same), NOT btrfs checksums (`+C`/nodatasum copies still 2.79×). Mechanism unknown -> needs blktrace/perf.
5. **Compression was a red herring.** /home is btrfs zstd:1; shards 33-83% encoded. But uncompressed `+C` copies (`/home/kmbandy/models/ds4-uncompressed/`, 151 GB) gave IDENTICAL 2.79× amplification and t/s. Fable's #1 rec (uncompress) did NOT help.
6. **DFlash drafter is EXCELLENT (76% accept, mean 3.97/4).** Spec-decode loses only on the 2.18× byte cost (batched verify blows up reuse distance 5×). It is a **win-in-waiting**: the moment effective bandwidth rises, spec-decode flips positive. Converter+runtime for DS4 DFlash all wired (Codex, commits 2b9666cfc/d9acf834b/0041d0f14). Draft model: `/home/kmbandy/models/dflash-speculator-DS4-Q8_0.gguf`.

## Why 1.84 (the bottleneck model — measured boundary, not a verdict)
Neither read path reaches the drive's ~6 GB/s in-engine:
- **Buffered**: 1.13× amplification (efficient) but random 4.45 MB buffered reads cap ~2.2 GB/s (page-cache overhead; standalone buffered io_uring = 1.1, WP gets 2.2 via IOSQE_ASYNC kernel-worker farm).
- **O_DIRECT**: reads at device speed BUT amplifies 2.79× in-engine -> 1.5 effective.
So ~2.2 GB/s effective -> ~1.84 t/s. The device's 6 GB/s is the headroom; both paths fail to reach it for DIFFERENT reasons. The 2.79× O_DIRECT amplification is the crack that, if closed, gives 5-6 GB/s (no amplification + O_DIRECT-fast).

## UNTRIED / UNRESOLVED LEVERS (prioritized)
1. **[LINCHPIN] Root-cause the 2.79× O_DIRECT amplification.** blktrace/blkparse + perf on the in-engine O_DIRECT reads vs the standalone microbench. Candidate causes: btrfs extent-read granularity on the specific expert offsets, block-layer read-ahead-equivalent, io_uring fixed-buffer interaction, RAID/LVM (none here), or a read-size bug. If O_DIRECT reads 1× at 5-6 GB/s in-engine -> ~3+ t/s AND flips spec-decode positive. HIGHEST VALUE.
2. **O_DIRECT direct-to-VRAM (P2P dma_buf, no host bounce).** The current O_DIRECT path host-bounces (read->pinned host->H2D). The "real" version reads O_DIRECT straight into the VRAM dma_buf window via align-down: read base=off&~511 for padded nbytes into slot, hand out slot_ptr+prefix (slot needs >=512B headroom). Avoids H2D entirely. Reuses the P2P ring's proven-async submit. Bigger change; may also dodge the amplification. (Codex flagged this as the next code route.)
3. **top-k expert pruning — UNTRIED, ready.** `--override-kv <arch>.expert_used_count=int:4` works at runtime (verified). Route top-4 instead of top-6 -> ~33% fewer expert bytes -> ~2.4 t/s territory. Quality is MEASURABLE: run wikitext PPL at top-6 vs top-5 vs top-4; if PPL barely moves, near-free. THE remaining byte-side software lever. Stacks with everything.
4. **Interleaved-GGUF repack (Fable #3).** On disk, an expert's gate/up/down are 3 separate tensors far apart -> 18 scattered 4.45 MB reads/layer. A converter that interleaves per expert -> 6 contiguous 13.3 MB reads/layer -> random->sequential (~7 GB/s regime). Offline repack, high value for bandwidth.
5. **Spec-decode compounding.** Banked (76% accept). Re-measure the instant bandwidth improves (lever 1/2). At effective 6 GB/s, spec's 2.18× bytes is paid for by 2.17× bandwidth -> ~2.3 t/s, more with top-k.
6. **Prefetch-overlap predictor (the realizable prediction path).** Instead of batched verify (which blows up reuse distance), go sequential decode + prefetch the drafted token's experts DURING current-token compute (keeps tight reuse, hides latency). Limited by autoregression: deep-layer routing needs the forward, so cheaply predictable only for hash layers 0-2 (12.5% of misses) via tid2eid. A learned cross-layer predictor (layer-L hidden -> layer-L+k experts, run on idle 6900XT) is a research bet.
7. **2nd NVMe (hardware).** Bytes are fixed + unhideable; raw bandwidth is the clean lever. But note: buffered is SOFTWARE-limited at 2.2 (< drive 6), so a 2nd drive only helps if lever 1/2 lets us actually use the bandwidth. Box has ONE x4 SN850X + a useless HDD.
8. **DSpark SAR / confidence-scheduled verify.** Cheaper verify (skip full MoE on high-confidence draft positions) -> cuts spec-decode's byte cost. Not in llama.cpp (feature request #25096). Out of scope unless upstream lands it.
9. **Batching** — REJECTED by user (single-stream only; helps throughput not latency).

## What to try NEXT (post-compact)
1. **blktrace the O_DIRECT 2.79× amplification** (lever 1) — the linchpin. Compare in-engine vs standalone read patterns at the block layer.
2. **top-k pruning A/B** (lever 3) — cheap, ready, measures quality. Do in parallel.
3. If bandwidth moves (1/2), **re-measure spec-decode** (lever 5) — it flips positive.
4. Consider **interleaved-GGUF repack** (lever 4) as a parallel bandwidth win.

## Artifacts / state
- **Harnesses** (`~/wp_logs/`): `odirect-run.sh` (O_DIRECT decode), `baseline-run.sh` (buffered), `spec-run.sh` (spec-decode, NMAX env), `odirect-uncomp.sh` (uncompressed). All: 6500 slots, hetero, 128-tok Roman-Empire prompt, self-manage load->request->shutdown.
- **Microbenches** (`/tmp/odirect_*.cpp`, hipcc -luring): dest/burst/fixed-file O_DIRECT tests. `~/wp_logs/wp_io_bench` (Grok's fio-like).
- **Analyzers** (`~/wp_logs/`): `analyze_v2.py` (eviction/Belady sim), `predictability.py` (miss-novelty).
- **Uncompressed copies**: `/home/kmbandy/models/ds4-uncompressed/` (151 GB, encoded=0) — DID NOT HELP, safe to delete.
- **Experimental O_DIRECT code** (uncommitted, behind `WP_ENSURE_BATCH_HOST=1`, default OFF -> does NOT affect the buffered default): worker pool + phase timers ("ODIRECT phase cum") + `WP_ODIRECT_PAGEABLE` gate in `src/weight-pager/wp-pager.{cpp,h}`. Snapshots for isolated diff: `~/wp_logs/pre-iofix-*`, `~/wp_logs/preodirect-*`.
- **Codex reports**: `~/wp_logs/codex-iofix-report.md` (submit-stall fix), `~/wp_logs/codex-odirect-report.md` (initial multi-QD), `~/wp_logs/codex-dflash-report.md` (DFlash).
- **Fable consult**: `~/wp_logs/fable-consult-report.md`.
- Working tree has unrelated in-progress changes from Grok (26+ files) — DO NOT assume clean; the buffered default path is untouched and is the 1.84 t/s working config.

## Process note
The device ground-truth (`/proc/diskstats`) repeatedly overturned WP's own metrics and my hypotheses (compression, pinned-mem, IOSQE_ASYNC all refuted by measurement). Rule for next session: MEASURE the block layer, do not extrapolate from WP counters or standalone benches. The 2.79× amplification is the one hard, reproducible, unexplained fact — chase it with blktrace before any more read-path code.
