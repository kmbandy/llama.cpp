# DFlash-Driven Predictive Expert-Streaming for DS4 Flash — Plan

Date: 2026-07-09. Branch: feat/wp-dflash-ds4. Goal: ~1.6 -> ~3.0-3.7 t/s (compute-bound ceiling).

## Measured ground truth (this session)
- Best transport = p2p BAR io_uring, 1.6 t/s, 1:1 device bytes (O_DIRECT dead-end: btrfs below-app 2.3x amplification).
- Per pass: 258 routed expert groups (6 used x 43 layers), 13.37 MB/group (MXFP4, 3x4.456MB sisters), 3.45 GB/pass.
- Pool: R9700 32GB, 6500 slots = 28.9 GB = ~8 passes headroom. Per token ~63% of experts change => ~1.02 GB/token cold NVMe.
- Compute/pass = 268 ms; I/O-wait = 358 ms; total 626 ms. SSD = 2.6 GB/s ACTIVE but ~40% duty (idle during the 268ms compute).
- Cross-layer routing signal REAL in TARGET residual: 0.64@top6, 0.82@top16 (1 layer ahead). Concentrated mid/late stack.
- DFlash lookahead: block_size 8, we run n_max 4, 76% accept, mean 3.97 tok => ~4 tok (~1s) reliable lead, up to 8.
- DFlash DIRECT projection through target routers = NEGATIVE (~random 0.017). DFlash hidden is its own space => needs ADAPTER.

## The ceiling (why the target is ~3-3.7, not 5)
Keeping up with compute = I/O-time-per-token drops to compute-time (268ms) => ~3.7 t/s compute-bound. Prediction's job is
to run the SSD at ~100% duty (fill the idle compute window) instead of 40%. Perfect overlap at 2.6 GB/s => ~2.9; push
transport toward 3.3+ => ~3.7. Beyond 3.7 needs a COMPUTE lever (kernel fusion) or fewer bytes (no more quant) - separate axis.

## Architecture: 3-tier predictive residency
- Tier 0 VRAM pool (~29 GB, ~8 passes): hot + prefetched experts. Predictor prefetches into it; scoreboard pins high-reuse.
- Tier 1 RAM victim (4-5 GB, slow-drain, ~28 GB/s): most-recently-evicted. Catches predictor mispredicts (20x cheaper than NVMe refill).
- Tier 2 NVMe (~2.6 GB/s active): cold + compulsory first-touch.
- Eviction demotes VRAM->RAM (not straight to NVMe); miss checks RAM before NVMe.
- Predictor (DFlash + adapter) drives prefetch. Confidence score = router_prob x token_acceptance x cross_pass_frequency.
- Progressive confidence-graded fill: near slots (high conf) dense, far slots (low conf) only high-conf shared experts.

## Tasks (build order)
### Task 1 [CRITICAL PATH] - DFlash->routing ADAPTER validation
Direct projection failed; test whether a small learned adapter recovers the signal. Offline, gates everything.
- Generate training pairs: (DFlash inp_g[pos], target routing[pos+1]) from captures (harness exists: capture-spec-run.sh + routing capture).
  Collect a few thousand positions across prompts for train/val split.
- Train a small adapter: per-tap-layer linear/low-rank [4096 -> target router logits] OR [4096 -> 4096 target-router-input] then reuse target routers.
  Start linear; add low-rank/MLP only if needed. Predict per target layer (or per tap, interpolate).
- Metric: cold-page recall vs over-fetch budget, at DFlash's lead (1-4 tokens). GATE: >=~0.6 recall @ modest over-fetch, calibrated
  (high-confidence predictions more accurate). If yes -> predictor viable. If no -> fall back to shorter-lead / transport-only (~2.9 ceiling).
### Task 2 [PARALLEL, low-risk] - RAM victim tier
Wire/verify WP_HOST_BUDGET_BYTES demote-on-evict + check-on-miss (mostly exists; measured -11.5% page_ins standalone). RAM-safety:
verify free RAM before pinning 4-5GB (15GB box, OOM kills desktop). Measure page_in reduction + refill latency in a predictive run.
### Task 3 - Prefetch engine (if Task 1 greenlights)
Residency manager with the 4 Sol interfaces (page-store / transport / residency / predictor). Confidence scoreboard over the
lookahead window; progressive fill; post-route submission to overlap. Pin high-cross-pass-frequency experts.
### Task 4 - Integrate + measure toward ~3 t/s. Isolated A/Bs, trusted baseline, no combined-lever runs.

## Parked / dead
- Smart eviction (dead: LRU ~optimal, Belady gap content-timed).
- O_DIRECT on btrfs (dead: below-app 2.3x amplification).
- Deeper io_uring queue (inert: QD 16=32=64 identical).
- >5 t/s without a compute lever (compute-bound at 3.7).

## Harness / artifacts (all in ~/wp_logs/accounting/)
capture-run.sh (target routing), capture-spec-run.sh (DFlash hidden), matrix-run.sh (transport matrix),
analyze-routing.py (target self-projection), dflash-align.py (DFlash->target), routing_capture.bin, dflash_capture.bin.
Code hooks (uncommitted, read-only, env-gated): wp-eval-cb.cpp WP_CAPTURE_ROUTING; speculative.cpp WP_CAPTURE_DFLASH.
P0 fixes (uncommitted): #5 DFlash-enable unified, #8 no silent wrong-expert substitution.
