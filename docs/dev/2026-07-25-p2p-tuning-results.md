> **SUPERSEDED IN PART — read `2026-07-25-final-pipelining-results.md` first.**
> Two recommendations below are WITHDRAWN on later measurement:
> 1. Window cache 4096 is called "the one real gain". It is **net harmful** — best read
>    bandwidth of any arm, worst throughput (1.817 vs ~2.73 tok/s). Reproduced twice.
> 2. The full-pool host map is named the highest-value remaining P2P change. **Withdrawn**
>    — it is the same mechanism amplified, so the evidence now predicts it is worse.
>
> Also: `tier_promotion_h2d_ms`, referenced later, has been deleted as unusable (its scope
> spanned the read window). The rest of this document stands — notably the 5.29 in-flight
> ceiling from ~6.05 pages per ensure_batch call, and queue depth / iowq being dead knobs.

# P2P tuning sweep — results. Target 5-6 GB/s NOT reached; one real gain, three nulls

**Date:** 2026-07-25 · **Machine:** mad-lab-main · **Tip:** `04ecae824`
**Goal (kmbandy):** P2P **and** the RAM tier working together on main at 5-6 GB/s, because
main must be the workhorse — mad-lab-2026's SN750 will only do 2.5-3 GB/s to RAM and has
no SAM/ReBAR.
**Logs:** `~/wp_logs/p2p/` (`summary.tsv` + per-arm), `~/wp_logs/p2p_{diag,sweep,dtd,iowq}.log`
**Harness:** `/tmp/p2p_ab.sh` (ARMS-driven; derived from `/tmp/task1_ab.sh`, same launch shape)

## 1. Verdict

**Target not reached.** Best P2P read bandwidth **3.293 GB/s** vs HOST's **4.39**. HOST
also wins on decode (3.140 vs P2P's best-reproducible ~2.84 tok/s). Of four knobs swept,
**one produced a real reproducible gain** (window cache), and three were nulls (queue
depth, iowq workers, direct-to-device-for-bandwidth).

Two structural constraints, both now measured rather than argued:

### 1a. In-flight depth is pinned at 5.29 — in ALL ELEVEN arms

`ensure_batch_p2p_inflight_avg_at_read_start = 5.29`, identical to 3 s.f. at queue depth
16 **and** 32, and across window caches from 64 to 4096. The cause:
**20025 pages / 3310 batches = 6.05 pages per `ensure_batch` call.** A decode step only
*has* ~6 pages to fetch, so no transport setting can raise concurrency.

`WP_P2P_QUEUE_DEPTH` is therefore a dead knob for decode. This also explains the
long-standing prefill/decode split: prefill sustains **6.206 GB/s** on the same drive and
the same code because its batches are large.

**The only way to raise in-flight depth is more pages per step** — cross-layer prefetch or
MTP speculation, not I/O tuning. Note the KG records xlayer prefetch as already measured
moving in-flight only 6.12 → 6.38, so that lever is known-weak in its present form.

### 1b. P2P's per-read cost is ~1.7x HOST's at identical concurrency

At 5.29 reads in flight both ways, for the same ~68 GB:

| | read_wait | implied per-read (4.25 MB) |
|---|---|---|
| HOST O_DIRECT pool | 13386 ms | ~4.9 ms |
| P2P best (window 4096) | 22498 ms | ~8.0 ms |
| P2P stock (window 64) | 26176 ms | ~9.3 ms |

To hit 6 GB/s at 5.29 concurrency each read must land in ~3.7 ms — i.e. **~25% faster than
HOST currently manages**, from a path currently 63% slower. Realistically needs both a
large per-read improvement and more pages in flight.

## 2. What each knob did

### WINDOW CACHE — the one real gain, and a threshold effect

| window | read_wait (ms) | io GB/s | n |
|---|---|---|---|
| 64 | 26176, 26411, 26640 | 2.837, 2.812, 2.788 | 3 |
| 256 | 27370 | 2.711 | 1 |
| 1024 | 26191 | 2.824 | 1 |
| **4096** | **22498, 22608, 22992, 23250, 23544** | **3.293, 3.277, 3.223, 3.186, 3.146** | 5 |

**−15% read_wait, +15% bandwidth at 4096, reproducible over 5 arms.** It is a THRESHOLD:
256 and 1024 are indistinguishable from 64. Cause — there are **5500 pool slots**, so 4096
windows covers ~74% of them and finally gets mapping reuse, where 1024 covers 19% and 256
just 5%.

**PROCESS NOTE:** I declared this hypothesis dead after seeing 256 and 1024 flat, then had
to retract when 4096 moved it. Both points were below the threshold. Sweeping two values of
a knob and concluding is the same error class as concluding from one run.

**Not maxed.** Codex's clamp caps the env at 4096; full coverage needs 5500. The code
deliberately avoids a "full-pool host map" (see the startup line). Mapping all 23375 MiB
once would remove per-read mapping entirely — **the single highest-value remaining P2P
change**, and the only one with direct evidence behind it. Read *why* the code avoids it
before assuming it is available.

### QUEUE DEPTH — null

16 vs 32: read_wait 26176 vs 26590, inflight 5.29 vs 5.29. See §1a. Note QD also *derives*
the default window cache (`max_windows_ = qd*4`, clamped [64,256]), so a bare QD32 arm
silently changes window cache to 128 — confounded by design, which is why window cache was
swept independently.

### IOWQ WORKERS — null, and verified honored

`WP_IOWQ_MAX_WORKERS` reached live code for the first time. Confirmed applied from the
startup line (`iowq_workers=8` / `32` / `kernel-default`), so this is a genuine null, not an
unapplied setting: read_wait 22498 (8) vs 23250 (32) vs 22992/23544 (default) — all within
4%.

### DIRECT-TO-DEVICE — no bandwidth effect, but the tier question answered

`WP_P2P_DIRECT_TO_DEVICE=1` at stock window: read_wait 26640, io 2.788 — indistinguishable
from baseline. **It buys no bandwidth on its own.**

Its value is correctness of the combination: **P2P and the RAM tier do work together.**
`hits` stayed **3135 in every single arm**, including direct-to-VRAM mode. The reason is
structural and I should have predicted it rather than worrying: this is a **victim** tier,
populated by VRAM evictions through `store_from_device`, *not* by fresh reads. So skipping
the fresh-read tier store costs nothing.

## 3. tok/s did not follow bandwidth — the fourth instance today

| config | io GB/s | tok/s |
|---|---|---|
| P2P baseline (w64, staging) | 2.81-2.84 | **2.835, 2.838** (0.1% spread) |
| P2P W4096 alone | 3.277 | 2.609 |
| P2P DTD+W4096 family | 3.15-3.29 | **2.219, 2.834, 3.075, 3.093** (39% spread) |
| HOST + tier | 4.39 | 3.140 |

A 15% reproducible bandwidth gain produced **no** tok/s gain. Worse, the tuned
configurations are dramatically *less* stable (39% spread) than the untuned baseline (0.1%).
An earlier reading of mine — "best P2P ≈ 2.955 tok/s" — was **retracted** once IOWQ32
returned 2.219 from the same family; the mean is ~2.805, i.e. no better than baseline.

This is the **fourth** read-path improvement today that did not convert to throughput
(zero-copy memcpy removal; −2528 ms tier read_wait; −15% P2P read_wait; +15% P2P
bandwidth). At ~6 pages per step the evidence is now strong that **decode is not
read-bandwidth-bound in this configuration.**

**Consequence:** establishing where decode wall-clock actually goes should precede any
further transport work. Four consecutive non-conversions is not a coincidence to keep
betting against.

## 4. Strategic note on P2P's role

P2P requires SAM/ReBAR, which mad-lab-2026 does not have — so **P2P can only ever run on
main, which is the box where it currently loses to HOST.** The case for it therefore rests
on its zero host-RAM footprint (no bounce buffers) rather than on speed, and that benefit
accrues to the machine that needs it least. Worth weighing before more investment.

What P2P does have: the **lowest read amplification of any transport** (64.98-68.77 GB vs
HOST 68.7, serial 72.33) and **no H2D stage at all**.

## 5. Session gains that are real

- Promotion on P2P: **999.5 → ~561 ms** (serial+copying → zero-copy borrow), holding across
  all 11 arms. Matches the ~470 ms predicted from the HOST comparison.
- P2P decode: 2.312 → ~2.836 tok/s (+23%), reproducible to 0.1%. **Only ~2 s of the ~10 s
  is attributable** (0.44 s promotion + ~1.6 s eliminated memcpy); the rest is reproducible
  and unexplained, because `promo_ms` times only `stage_in` and never covered the old
  `lookup()` copy.
- P2P + RAM tier co-residency: proven working, tier hit rate unaffected.

## 6. Recommended next steps, in order

1. **Decode wall-clock attribution.** Gates everything else. Four non-conversions.
2. **Full-pool host map for P2P** (Codex). The one evidenced bandwidth lever; also raise or
   remove the 4096 clamp. Expect bandwidth, do NOT expect tok/s.
3. **More pages per step** (cross-layer prefetch / MTP) — the only route to >5.29 in flight,
   and therefore the only route to 5-6 GB/s. Known-weak today.
4. **Amplification attribution** — needs the debug alignment override. `~/models/ds4_zstd`
   (149 GB) exists for this and **must be deleted** afterwards.
5. **The distributed split** — the stated first priority, still untouched.

## 7. Defaults recommendation

Keep the shipped defaults. `window_cache=4096` is a genuine +15% bandwidth for zero risk
and could reasonably become the default, **but** every arm using it showed markedly worse
tok/s stability, so it should not be made default until §3 is understood.
