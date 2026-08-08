# Weight pager / DS4 spec prefetch — evening handoff

**Written** 2026-08-06 ~23:40, end of the evening session that executed
`2026-08-06-weight-pager-handoff.md`. **Author** Claude (outgoing). Full
run-by-run data: `2026-08-06-all-runs.txt` (evening sections). Continuation
brief `d83d15a1` auto-injects next session; KG anchors listed at the bottom.

## Targets (kmbandy, tonight)

**Prefill 80–100 t/s. Decode 10–20 t/s — 10 without top-k or pruning is
acceptable.** Top-k 6→4: skeptical (GLM precedent). Pruning: only ever via
existing REAP builds; full model preferred. Baseline as of tonight: ~22–24
prefill / ~3.3 decode, config of record, verified reproducing.

## What tonight fixed (all committed, ff03d11d0..e27257106, UNPUSHED)

1. **RX 480 runtime PM** was evacuating all VRAM to system RAM on every ≥5s
   idle gap — the §5.1 "GTT migration", the 7.5 GB RAM spikes, the third leg
   of the 14:56 OOM, and the reason KEEPALIVE=100 was worth +17.5%. Fixed by
   kmbandy: `power/control=on`. **THE UDEV RULE IS STILL PENDING — this
   reverts on reboot.** Rule text is in the brief and the board announcement.
2. **The host victim tier served wrong bytes on ~every Vulkan restore**
   (604/605): the DeviceReader matched slots by `tensor->data`, which is a
   colliding sentinel on Vulkan. Fixed in bb0654986 (reader keys on
   cache_id); proven 0/607 post-fix. This was why tier-on runs generated
   divergent text and decaying acceptance. **The rig is now byte-deterministic**
   — identical configs reproduce text and draft trajectories exactly.
3. **WP_EXPERT_SPEC_LEASE=256** (vs default 64): halves R9700 LATE 50→23%,
   used 47→75%. Found by offline sim (`sim-lease.py`), confirmed live.
4. Tier sizing: **3 GiB/worker safe on both boxes** (2026 floor 2.9 GB);
   6 GiB on main is a strong NEGATIVE (reclaim + per-hit copies; 1.15 t/s).
5. **SPEC_NMAX=7 + SPEC_CONF=0.6 flips spec-on decode positive**: 3.29/3.31
   vs 3.26/3.27 off vs 2.77–3.17 at conf 0.99. Mechanism measured: more
   draft attempts → more tokens per verify sweep → R9700 requests −5%,
   page-ins −4.5%, wait −5.5%. Not yet the record: 0.99 arms show a 0.40
   spread across identical trajectories that the ladder must explain.

## The plan to the targets (morning order)

1. udev rule + sync 2026 (2 commits behind: 9d473814d, e27257106).
2. **Conf ladder**: 0.99/0.9/0.6/0.4 × nmax7, ≥4 interleaved reps,
   per-worker waits. Metrics: R9700 page-ins/token + wait, not tok/s.
3. **The prediction build**: dispatch-side per-layer emission —
   `router_{L+1..L+k}(h_L)` (k=2–4) over all verify rows, top-M=2..3,
   union, PREDICTED provenance, SPEC_HOST=1 landing. Day-one falsifier:
   the L+k precision decay curve (k=1 is 97.3% rank-1; k≥2 never measured).
   Key frame (kmbandy's, correct): **Belady bounds miss COUNT, not
   latency-hiding** — a read issued 10ms early costs idle bandwidth
   (drive is 60–70% idle), not critical-path time.
4. **Prefill streaming spike**: sweep demand is ~dense ⇒ deterministic ⇒
   stream layer L+1's shard while L computes. Ceiling ~170 t/s marginal;
   the target is under it. First: gate speculative reads OUT of prefill
   (n_tokens>1 requests; prefill spec LATE is 84–100% = pure contention).
5. **turbo4 re-adjudication**: 1.96× decode / 1.49× prefill measured, parked
   on nondeterminism evidence that predates the corruption fix. One temp-0
   byte A/B decides it. 3.3 × read-hiding × turbo4 ≈ 10–11.6 — the target
   without touching quality.

## Measured design constants (do not re-derive)

- Verify-batch union economics: 6.76 → 5.61 → 4.63 → 3.99 experts/layer/token
  at n = 1/2/4/7 (−41% at depth 7). Adjacent-token overlap 0.33–0.42
  (≤1-token-ahead prediction stays dead). Sweep carryover at n=7: 0.455.
- Cross-layer (2026-07-19, control 0.98): rank1 0.973, rank2 0.914,
  rank3 0.814; M=2–3 pre-stages 31–45% of the next layer at 1.02–1.05×
  bytes. The old "regressed" verdict came from M=16–48.
- Decode residency on plain LRU: 72–73%. NVMe idle 60–70% of decode.
- LRU ≈ best online policy (2Q ties, FIFO/LFU far worse); OPT gap 34–38%.

## Traps added tonight (beyond the morning handoff's list)

- `routing_capture.bin` was OVERWRITten by a GLM run (fopen("w")) — the July
  DS4 hiddens are gone; WP_CAPTURE_ROUTING lives on the pager path only and
  cannot fire in dispatch mode. The build's emission site is the new capture.
- Killing a local background task does not kill the remote harness; a stray
  arm raced its replacement tonight. Verify by explicit PID on BOTH boxes.
- Multi-run chains: setsid-detach on 2026 + Monitor the log; the Bash tool
  caps at 10 minutes.
- WP_EXPERT_TIER_VERIFY=1 is diagnostic-only — never in a measured arm.
- Board claims are per-campaign-block; two arms ran claimless tonight — don't.
- Read tier hit counts WITH wait (hits-up + wait-up = the tier losing).

## KG anchors

9171cc3b (runpm root cause) · 97657d4d (tier corruption) · 8ffaa587
(build basis) · 2299e328 (targets + map) · 6b6b60a4 / 25da4858 (session
summaries) · brief d83d15a1. Standing constraints unchanged from the
morning handoff §11, including: no re-sharding; non-Claude subagents never
run GPU/inference; live services untouchable.
