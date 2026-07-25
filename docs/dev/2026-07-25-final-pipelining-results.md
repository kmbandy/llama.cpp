# 2026-07-25 final: pipelined promotion SHIPS, window-cache tuning RETRACTED

**Machine:** mad-lab-main · **Repo:** `~/GitHub/llama.cpp` master · **Tip:** `4fae73830`
**Supersedes the recommendations in** `docs/dev/2026-07-25-p2p-tuning-results.md` — see §3.
**Logs:** `~/wp_logs/final_run.log`, `~/wp_logs/p4/summary.tsv`

## 1. Pipelined tier promotion: measured win, now DEFAULT ON

Clean-box A/B, P2P + 4 GB tier, 3 rounds with **order alternated each round**:

| round | first arm | second arm | ON delta |
|---|---|---|---|
| 1 | OFF 2.603 | ON **2.730** | +4.9% |
| 2 | ON **2.773** | OFF 2.741 | +1.2% |
| 3 | OFF 2.621 | ON **2.699** | +3.0% |

**ON wins all three, in both orders.** Means: ON 2.734 vs OFF 2.655 = **+3.0%**.

The mechanism, and this is the part that is solid rather than suggestive — both arms
replicate to ~2% and the gap is far outside that:

| | OFF | ON |
|---|---|---|
| `jobs` ms | 274.7 / 280.2 / 276.8 | **4.4 / 4.5 / 4.5** |
| `promo` → `fence` ms | 565.0 / 575.6 / 567.0 | **277.3 / 277.0 / 275.7** |
| **total promotion overhead** | **846.4 mean** | **281.1 mean** |

**−565 ms.** `WP_PIPELINE_PROMOTIONS` now **defaults ON** (`4fae73830`); opt out with
`=0`, which remains the A/B control.

**Why this one converted when four others didn't.** Four earlier read-path
improvements today produced no throughput change (zero-copy memcpy removal, −2528 ms
tier read_wait, −15% P2P read_wait, +15% P2P bandwidth). Those all shrank phase time
that was **already overlapped with compute**. This change is different in kind: it
removes a **serialization that blocked NVMe submission** — promotions no longer have
to complete before the cold-miss batch can be submitted. It does not make I/O faster;
it stops promotion from delaying I/O. That distinction is the useful generalisation
from the whole day.

**NOT claimed:** a read_wait improvement. OFF spans 26993-28827 ms (6.8%) and ON
27165-27935; the ranges overlap and OFF's own variance exceeds the difference. An
earlier revision of this analysis claimed −994 ms; retracted.

## 2. Window cache 4096: RETRACTED, it is net harmful

`WP_P2P_WINDOW_CACHE_MAX=4096` produces **the best read bandwidth and the worst
throughput of any arm:**

| | read_wait | io GB/s | tok/s |
|---|---|---|---|
| ON (stock window 64) | 27165-27935 | 2.66-2.73 | **~2.73** |
| ON + window 4096 | **25858** (lowest anywhere) | **2.893** (highest) | **1.817** |

Measured phases for the 4096 arm total 26.6 s versus 28.6 s for stock — better on
everything instrumented — while wall-clock is 70.4 s versus 47.4 s. The cost is
outside every counter, and the plausible mechanism is the host-side price of ~4096
mapped 4.25 MB windows (~17 GB of VA mappings and the attendant TLB pressure).

**This reproduces.** An earlier quiet-box arm also showed better bandwidth (3.277
GB/s) with worse throughput (2.609 vs 2.838). The direction is consistent across two
days and two load conditions. The **magnitude** from the second measurement is less
certain (load rose during that sweep), but nothing supports window 4096 being useful.

## 3. Corrections to `2026-07-25-p2p-tuning-results.md`

That document is wrong in two specific ways and its recommendations should not be
followed:

1. It calls window cache 4096 **"the one real gain"**. It is not a gain. It trades
   ~15% read bandwidth for a much larger loss elsewhere. Struck.
2. It names **the full-pool host map as "the single highest-value remaining P2P
   change"**, reasoning from the window-cache threshold effect. That recommendation is
   **withdrawn** — the full-pool map is the same mechanism amplified (map the entire
   23375 MiB pool instead of 17 GB of it), so the evidence now predicts it would be
   worse, not better.

The rest of that document stands, in particular: in-flight depth pinned at 5.29 across
every arm because decode fetches ~6.05 pages per `ensure_batch` call; queue depth and
iowq workers are dead knobs; P2P and the RAM tier coexist correctly because the tier is
a victim tier fed by evictions rather than fresh reads.

## 4. The crash that this work exposed

Worth recording because the sequence is instructive. A "3× regression" was reported
from arms measured while another user's container build saturated the CPUs. On that
false premise a gate (`0d0422fac`) was added to protect master. **The gate's
"restore the pre-pipeline path" branch dropped a `borrow()`**, leaving
`host_hit_src[k]` null and passing that null to `hipMemcpyAsync` — a deterministic
abort in HOST + tier (`ROCm error: invalid argument`, surfacing at the next
`get_rows_cuda` launch because the error was latched earlier).

So a gate added to defend against a regression that did not exist introduced a real
crash into the configuration we actually ship. It was found only by going looking for
damage from the non-existent problem. Fixed in `ae2595955` (reacquire the borrow, hold
the handle in `HostBorrowGuard` through `hipDeviceSynchronize`) and GPU-verified: HOST
+ tier COHERENT in both flag modes, flag-off reproducing pre-pipelining counters to
within 1%.

Localisation that made it findable: HOST + tier crashed, HOST without tier was clean,
P2P + tier was clean having driven 3135 promotions through the same `page_in_sync_`.

## 5. Instrumentation removed

`tier_promotion_h2d_ms` is **deleted** (`4fae73830`). It reported values nearly equal
to that arm's `read_wait` in both modes (28277 vs 27712; 27184 vs 27165) because its
scope spanned the read window. It was quoted in a report before the error was caught. A
counter that prints a plausible wrong number is worse than no counter.

`tier_promotion_fence_ms` is kept and is trustworthy — a stable 275-281 ms, and the
counter that made the win visible.

## 6. Measurement practice, for the next session

Everything that survived today was measured this way; everything retracted was not.

- **Alternate arm order every round** and take the within-round comparison. This is
  what makes a result robust on a shared workstation that is doing other things —
  which is the normal condition, not an aberration to be waited out.
- **Phase counters are the primary signal; tok/s is directional.** Counters replicate
  to ~1-2%; tok/s swings 8-17% on identical configurations.
- **n=3 minimum.** n=1 and n=2 each produced a retraction today.
- **Never compare arms across sessions.** Two retractions came from exactly that.
- **Bandwidth metrics can move opposite to total time.** `io_gb_s` and read_wait
  improved while wall-clock got worse in three separate instances. Optimise time.
- **Check the box state at both ends of a sweep**, not just the start.
- **Build the full target set.** A single-target build left a day-old `llama-server`
  while `libllama.so` was current; object timestamps for one file proved nothing about
  the binary under test.

## 7. Open

1. **The distributed split** (roadmap P2) — the stated first priority, still untouched.
   8 GB RAM tier per machine is settled. Blocker remains structural: no pipeline
   parallelism in llama.cpp and RPC cannot express it.
2. **Amplification attribution** — the 2.49x is still unattributed between the 512-vs-4096
   alignment fix and btrfs zstd. The compressed test copy has been **deleted** (it was
   the wrong experiment: the model is incompressible outside shard 4, so compression
   cannot explain a uniform 2.49x). The right test is a debug override forcing alignment
   back to 512 on the uncompressed file — one variable, no extra disk. Only worth doing
   if the answer would change a decision about the second machine.
3. **Event-pool exhaustion fallback: untested.** The fault injection was misdesigned —
   shrinking the pool cannot starve it while the fence is per-batch, so at most ~6
   promotions are ever outstanding. Not reachable without a GPU integration harness.
4. **Prefetch** is the only route past the 5.29 in-flight ceiling, and is the thing the
   dual-path architecture's payoff is contingent on. Currently known-weak (6.12 → 6.38).

## 8. Architecture decision (kmbandy, settled)

The dual-path route is **in**: RAM→VRAM for warm/prefetched pages, NVMe→VRAM direct for
cold misses, concurrently. Rationale is host-RAM capacity and bandwidth on the 8 GB
per-machine budget — miss traffic stops costing ~136 GB of RAM traffic per run and stops
polluting the tier — not raw throughput. P2P remains ~10% behind HOST on decode today
(~2.73 vs ~3.0-3.1) and only runs where SAM/ReBAR exists, i.e. main and not
mad-lab-2026. Do not re-litigate; further validation was explicitly declined.
