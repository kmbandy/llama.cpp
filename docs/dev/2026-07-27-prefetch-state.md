# Cross-layer prefetch — state of play, 2026-07-27

**Headline: prefetch has still never been correctly tested.** Four gates blocked
it; three are now fixed and the fourth is a correctness bug that was *invisible
until the other three were lifted*. Every prior "prefetch is dead" verdict was
measuring a broken implementation, not the technique.

`WP_PREFETCH_FIX` is default **OFF** and must stay off until gate 4 is fixed.

---

## 1. The four gates

| # | gate | why it blocked | status |
|---|---|---|---|
| 1 | **scheduler queue** — `wp-pager.cpp:886` returns on `free_queue_slots() <= 0` | capacity held by Done-but-unreaped entries no reservation can free | FIXED (`spec_reap_`) |
| 2 | **pool free slots** — `prefetch_pages_batch(allow_evict=false)` caps at `n_free_unpinned()` | permanently 0 on a warm pool, so speculation never allocates, so no slot ever becomes speculative | FIXED (`af1778211`) |
| 3 | **harvest promotes** — `harvest_ready_prefetches_` called `mark_used()`, which clears `speculative_` | prefetched pages promoted to the hot set the instant their read landed, so the tier never accumulated and eviction fell onto DEMAND | FIXED (`cbdcee4f8`, new `PoolAllocator::touch_lru()`) |
| 4 | **CORRECTNESS RACE** | speculation corrupts output as soon as it actually runs | **OPEN** |

All three fixes ship behind one switch (`WP_PREFETCH_FIX=1`) because **none works
alone**: 1 without 2 submits nothing, 1+2 without 3 evicts the demand working set.

## 2. Measured, laguna, 3400 slots (~12% residency), 2026-07-27

```
      gen_hash        tok   page_ins  io_gb_read   t/s
CTL   3e55091e46db     22     57267    109.181    1.70
FIX0  3e55091e46db     22     57267    109.181    1.71   <- fix on, xlayer off
FM2   8fb9645622b2    128     50986     96.579    3.40   <- CORRUPT
FM4   8fb9645622b2    128     52329     99.053    3.24   <- CORRUPT
CTL2  3e55091e46db     22     57267    109.181    1.74

FM2 [xlayer] blocked_free_queue=196 bootstrap=5211 submitted=5385
             hit=10  spec_evict_unused=5171  n_spec=1
```

**Gates lift:** `submitted` 4 → 5,385. `bootstrap` 0 → 5,211.
`blocked_free_queue` 6,207 → 196.

**`FIX0` is byte-identical to control** on generation, token count, `page_ins`
and `io_gb_read` — the machinery is inert when speculation is off.

**And it corrupts.** Controls stop at 22 tokens with `The capital of France is
**Paris**.`; the prefetch arms run to the 128-token cap and emit **no printable
text at all**. The 2× throughput is the 07-21 fake-win signature — degenerate
routing pages with trivial locality and looks fast.

## 3. What this corrects in the record

- **The 07-21 race is NOT "K>=2 with M>=8".** It was filed that way because low-M
  arms looked clean — but they looked clean because **nothing was ever
  submitted**. There was no race to fire. It corrupts at **M=2** now that
  submissions happen. Every "M is too wide" conclusion was measuring the gates.
- **Four knobs previously filed DEAD were all downstream of gates 1–3:**
  `WP_PREFETCH_MAX_SLOTS` (`blocked_budget=0` because `n_speculative()` is
  always 0), `WP_SPEC_RESERVE` (reserves *queue* slots, not *pool* slots),
  `WP_PREFETCH_DEPTH`, and the io-wq worker cap. None could bind.
- **`WP_SPEC_REAP`'s apparent "unblocking" (18 → 41,778)** was it bootstrapping
  the tier as a side effect — harvest commits finished prefetches and releases
  their slots — while racing harvest against use. That is likely the 07-21
  corruption, and gate 3 is the half of it that was never identified.

## 4. Next diagnostic (named, runnable, not yet run)

**`WP_SPEC_BOOTSTRAP=1 WP_SPEC_KEEP_TIER=1 WP_SPEC_REAP=0`** with xlayer on.
Localizes the race:
- still corrupt → it is in the **fetch-into-slot** path
- clean → it is in **harvest/eviction**, and reap is the trigger

This has been the named next step since 07-21 and is only now meaningful,
because before this it would have measured a path that submitted nothing.

## 5. Separate concern, independent of the race

`hit = 10` of `submitted = 5,385` = **0.19%**. `spec_evict_unused = 5,171` =
**96% of speculation thrown away unused.**

Offline analysis predicted far better (precision@rank1 = 0.973; budgeted
prefetch at M=2–4 pre-staging 31–56% at ~1.0–1.10× bytes). Either the online
predictor is not the offline one, the lookahead is wrong, or the corruption is
poisoning the routing that drives the prediction. **Do not invest further in
prefetch tuning until this is reconciled** — even a correct implementation may
not earn its bytes at this hit rate.

## 6. Correctness gate — the method that works (hard-won)

Four broken comparators in one session. The procedure that survives:

1. **No `--ignore-eos`.** It forces generation past EOS into a degenerate loop
   where near-tied logits make FP noise flip the tail. `--ignore-eos` and
   text-identity gates are **mutually exclusive**. Use it for throughput/byte
   statistics with a fixed token budget; use natural EOS when comparing output.
2. **Separate stdout from stderr.** Generation is stdout; diagnostics are
   stderr. `2>&1` puts `page_ins` in your hash.
3. **Compare generation only** — the tail of the filtered stdout, clear of the
   banner and spinner, whose length varies with load time.
4. **Self-test the comparator** on known-same and known-*appended* inputs before
   trusting it. (A *prepended* diff does not test a tail-based comparator — that
   self-test passed vacuously today.)
5. **Carry an independent signal.** The **token count** from stderr caught this
   corruption when the hash comparator was broken. One instrument is not enough.

Verified baseline: `gen_hash 3e55091e46db`, **22 tokens**, `The capital of
France is **Paris**.`

## 7. Why this still matters for Kimi-K3

K3 runs at ~12% residency (~25% hit) where a working prefetcher has far more to
do than laguna at 88%. The predictor's offline precision@1 of 0.973 is real. The
technique remains **untested**, not refuted — which is the same trap logged
twice already (the weight-pager's ten arms at `LOOKAHEAD_K=1`, and the DSWS
prefetch built with no lead time). Do not retire prefetching on the strength of
a broken build.
