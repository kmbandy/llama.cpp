# Morning brief — 2026-07-31

Cross-machine expert dispatch, GLM-5.2. Written end of 2026-07-30, ~23:00.

**Read §1 and §2 first.** Everything else is reference. Nothing in here needs
re-deriving — it is all measured, and where it is not measured it says so explicitly.

---

## 0. One-paragraph state of play

The system works. Four GPUs across two machines, three backends, coherent output,
**0.941 tok/s**, and a worker dying no longer kills the spine. The whole decode is
dispatch wait (99.2%), and we now have that wait fully decomposed down to individual
legs. The single dominant cost is **the H2D upload of missed expert pages, running at
~1.29 GB/s — roughly a tenth of what the bus should do.** That is 60% of the budget and
is the immediate target. Expert *arithmetic* is 4% of the budget; do not spend time
there.

---

## 1. IMMEDIATE GOAL — the H2D upload path

### 1.1 The number

Measured per-request budget on the GTX 1070, which is the fleet's pace-setter
(per-layer wait is set by the slowest worker):

| leg | ms | share |
|---|---|---|
| arithmetic | 0.54 | **4%** |
| **H2D upload of misses** | **7.24** | **60%** |
| NVMe read | 4.36 | 36% |
| **total** | **12.1** | (observed wait 11.2–12.5 ms — budget reconciles) |

The upload figure comes from solving across two runs with very different miss rates:

```
4.89 ms/expert = base + 0.482 * U      (normal run, 48.2% miss)
0.59 ms/expert = base + 0.027 * U      (degenerate run, 2.7% miss)
  =>  U ~= 9.45 ms per MISS,  base ~= 0.34 ms per expert
```

**9.45 ms to move one 12.2 MB expert page = ~1.29 GB/s.** PCIe 3.0 x16 delivers
10–12 GB/s with pinned memory and 3–6 GB/s pageable. We are at roughly a tenth of the
bus.

### 1.2 What to actually do

**Step 1 — find out WHY it is 1.29 GB/s before assuming it is pinning.** Candidates, in
order of suspicion:

1. **Unpinned (pageable) staging memory.** The staging arena is
   `posix_memalign`-allocated (`wp-expert-worker.cpp` ~line 562), i.e. ordinary pageable
   host memory. Pageable H2D forces the driver to bounce through an internal pinned
   buffer. This is the most likely cause and the cheapest fix.
2. **A double copy.** The page may land in the staging arena and then be copied again
   into a device-visible buffer rather than uploaded directly. Read the path from
   `read_page` through to the slot write and count the copies.
3. **Synchronous per-page upload** with no batching or overlap, so every page pays full
   latency with no pipelining.

**Do not skip step 1.** I assumed "pinning" in conversation; it is not verified. If it is
actually a double copy, pinning buys nothing.

**Step 2 — the fixes, in order:**
- pinned/page-locked staging buffers (`hipHostMalloc` / `cudaHostAlloc` equivalents —
  but see the backend warning in §6.3, this must be backend-neutral)
- async copies on a separate stream so upload overlaps NVMe read and compute
- batch multiple page uploads per submission

**Step 3 — required instrumentation.** `ns_h2d` is currently reported as `unavailable`
because there is no independent completion boundary before the existing readback. To
measure this properly we need a real `ns_h2d`. Getting one may require a diagnostic-only
sync — that is acceptable **if and only if** it is behind its own flag, clearly labelled
as perturbing, and never used to quote throughput. See §5.4 for why this matters.

### 1.3 Expected payoff

1.29 → 6 GB/s takes the layer from 12.1 ms to 6.46 ms: **0.94 → ~2.1 tok/s**, before any
other lever. Overlapping upload with the NVMe read takes it to ~4.9 ms (**~2.7 tok/s**).

---

## 2. Verified state — what works and what it cost

All measured 2026-07-30, all reproducible with the harnesses in §7.

- **4 GPUs / 3 backends / 2 machines, coherent output, 0.941 tok/s.**
  Spine = RX 6900 XT (dense resident, 14.298 GiB, on TB3). Workers = R9700
  (experts 85–255), GTX 1070 (CUDA) and RX 480 (Vulkan) both holding 0–84 and
  self-partitioning by residency affinity.
- **Worker loss no longer kills the spine.** Verified 3 times end-to-end by SIGKILLing a
  worker mid-generation: spine alive, `/health` ok, clean HTTP 500, log names the dead
  endpoint and the in-flight expert ids. It also caught a real regression tonight, not
  just a synthetic test.
- **Decode is 99.2% dispatch wait.** `pack 0.34 / issue 4.4-5.0 / wait ~900 / unpack 0.18 ms`
  per token across 75 routed layers.
- **All three workers genuinely serve.** Proven from `/proc/PID/io` read_bytes and CPU
  deltas, not log greps — the worker prints nothing per request, so grepping for
  "dispatch" returned 0 on every run regardless of truth.
- **RX 480 pulls its weight.** Removing it drops throughput 0.889 → 0.727 tok/s (~20%).
- **No read amplification.** MB/miss is 11.63–11.66 on all three workers against a
  ~12.2 MB page. The btrfs `compress=zstd` / O_DIRECT pathology that once cost 2.49×
  has not returned.
- **Cache coverage is NOT the problem.** Miss rates 46% / 53% / 49% are near-identical
  despite the R9700 having 15.6% slot coverage vs ~7.8% per 2026 worker.

### 2.1 Hypotheses that are DEAD — do not revisit

| hypothesis | verdict | evidence |
|---|---|---|
| per-layer scheduler barriers / GPU pipeline drains | **refuted** | dispatch `total` ≈ entire eval time; the gap outside the op is ~0 |
| cross-worker serialization | **refuted** | `first_await_in_flight` = 2.6–2.7 of 3; requests are issued before awaiting |
| cache coverage / slot count | **refuted** | miss rates equal across very different coverage |
| NVMe contention as the primary cause | **refuted** | giving the 1070 the SN550 alone improved per-expert only ~20%, not the ~5× gap |
| expert arithmetic / GPU generation | **mostly refuted** | arithmetic is 4% of the budget; 0.34 ms/expert, ~7× above roofline |
| "2026 workers want MORE experts per request to amortize fixed cost" | **refuted** | fixed is 1.96 ms/req vs 3.93 ms/expert marginal — marginal dominates |

---

## 3. The budget and the ladder

Derived from the measured legs in §1.1. **This is a prioritization order, not a
forecast** — the shares are measured, the individual multipliers are estimates.

```
today                              12.10 ms/layer ->  0.94 tok/s
pinned/async H2D (1.29 -> 6 GB/s)   6.46          ->  2.06
+ overlap upload with NVMe read     4.90          ->  2.72
+ prefetch (miss 48% -> 26%)        2.90          ->  4.61
+ MTP speculation ~1.5x                           ->  6.91
+ expert deferral hiding ~half                    -> 11.05
```

These largely compose rather than overlap: prefetch cuts the *number* of misses, pinning
cuts the *cost* of each, deferral hides what remains, MTP multiplies tokens per step.

### 3.1 The ceiling

Only the two 2026 workers cross the wire; the R9700 worker is loopback on the spine's own
machine.

```
2 remote workers, 104 MB/s measured link:
  72 KB/layer  ->  0.68 ms/layer  ->  19.7 tok/s ABSOLUTE CEILING
  with F16 downlink (24.6 -> 12.3 KB):  48 KB/layer  ->  29.6 tok/s
```

**The F16 downlink is cheap and should be banked early.** Activations already go out as
F16; the partial sums come back F32 for no strong reason. At 15 tok/s the wire alone
would be ~80% of the token budget without it.

kmbandy's stated goal: if we get near 29.6 tok/s, pull Kimi (Q1) — plausibly ~10 tok/s
at that scale.

---

## 4. Next levers, in order

1. **H2D upload path** — §1. 60% of the budget, running at a tenth of the bus. Nothing
   else is close.
2. **F16 downlink** — cheap, raises the ceiling 19.7 → 29.6 tok/s, needed before 10+ is
   reachable at all.
3. **Prefetch policy** — plumbing exists, *no policy*. Each miss now costs both the NVMe
   read AND the ~9.45 ms upload, so miss reduction pays twice. **Measure GLM-5.2's
   routing predictability first** — the promising numbers in the KG (top-3 prefetch,
   +5% bytes, 45% pre-staged) are from DS4, a different model with 6 experts/token vs
   GLM's 8 of 256.
4. **MTP speculation** — GLM has the head, but `blk.78.nextn.*` is currently in the
   "unused... ignoring" list and is not wired up at all. Real work, not a flag.
5. **Expert deferral (KTransformers, SOSP'25)** — defer ~6 of 8 experts, overlap with the
   next layer's attention. The published recipe is tuned for single-node DDR5; our
   accounting differs (gigabit, 0.5–0.6 ms RTT, three heterogeneous workers) and must be
   re-derived, not copied.
6. **Capability-weighted re-shard** — worth ~17% on the compute leg only, and needs a
   re-shard (2026 down to ~0–76 from 0–84). Low priority now that arithmetic is known to
   be 4%.
7. **LFRU eviction + learned hotlist warm-start** — from the waste survey. Hit-rate
   levers, cheap A/B, sequence after the above.

Also logged as explore-targets (decision `291cec4b`): MoE-Infinity (trace-driven
prefetch), FlashMoE (learned replacement).

---

## 5. Mistakes made tonight — read this, it will save you hours

### 5.1 The `-ot` regex is an unanchored substring match
`std::regex_search` at `src/llama-model-loader.cpp:1231`. `-ot 'output\.weight=CPU'`
matched every `blk.N.attn_output.weight` too — **6.212 GiB moved off the GPU**, all 79
attention output projections running on CPU. Use `token_embd\.weight` alone (unique, and
it is only a gather so CPU is genuinely cheap). Verify placement from the server's own
`[load_loop] ctx[N] ... buft=` lines, never from `rocm-smi` totals.

### 5.2 A stale spine silently hijacks the measurement
A leftover `llama-server` holding port 8095 answered the harness's `/health` probe while
the run's *own* spine had died with "couldn't bind". The run reported success while
measuring a completely different configuration. That is where a bogus 0.593 tok/s came
from. Both `llama-server` and the worker **survive SIGINT**, so teardown must escalate to
SIGKILL. The harnesses now refuse to start if the port is bound and abort if the spine
PID dies.

### 5.3 Output sha256 is NOT a config-change detector
Output was **bit-identical** before and after moving 6 GiB of weights between devices,
because greedy argmax masks small numeric differences. Verify changes directly.

### 5.4 The confound that cost the most: comparing runs at different cache states
I reported an "8.7× compute speedup and 1.455 tok/s" from graph batching. **Both were
artifacts.** The batched build shipped with a broadcast bug that corrupted prefill,
which collapsed expert routing onto a tiny set and drove the miss rate to **2.7%**.
Matched at ~48% miss, batching is worth **~6%** (5.19 → 4.89 ms/expert).

The lesson is sharper than "be careful": **my mechanism counters did not save me.**
`n_graph_submits` and `n_device_allocs` correctly proved the batching *engaged* — they
said nothing about whether it *caused* the delta. A mechanism counter proves the
mechanism ran; it does not control for a confounding variable. **Always check miss rate
before attributing any throughput change.**

### 5.5 A per-request aggregate is not a hardware comparison
I reported "1070 is only 1.24–1.32× slower per request" and concluded fixed cost
dominated. Wrong: those workers carried 3.4× fewer experts per request. Varying the load
on the *same* worker gave `M = 3.93 ms/expert, F = 1.96 ms/request` — marginal dominates.
When two things differ in both rate and workload, vary the workload and fit the line.

### 5.6 The ggml broadcast trap (this cost the correctness bug)
`ggml_mul` broadcasts via `ggml_can_repeat`, which only checks `ne[i] % src1->ne[i] == 0`.
A 1-D `[n_tokens]` routing-weight tensor against a `[n_embd, n_tokens]` output **passes**
that check (`6144 % 8 == 0`) and then scales along the **embedding** axis. Silently wrong,
no assert. Correct only at `n_tokens == 1`, so decode looked fine while prefill was
corrupted. Fixed by shaping it `[1, n_tokens]` (`ggml_new_tensor_2d(ctx, F32, 1, n_tokens)`),
with a comment at the site.
**A single-token unit test cannot catch this** — at `n_tokens=1` both readings coincide.
Any test of per-token weighting must use `n_tokens >= 2` with *distinct* weights.

### 5.7 Incomplete build target lists cause ABI crashes
Inserting `uint64_t host_victim_bytes` into `struct Options` ahead of `TestHooks *`
changed the layout. The library was rebuilt; the **executable was not**, because every
handoff said only "rebuild libllama and llama-server". The library then read `test_hooks`
from where main wrote `once` — garbage non-null pointer, called on the first request,
segfault on all three backends. Unit tests passed (a self-consistent test build has no
ABI skew).
**Rules:** enumerate every target explicitly; prefer appending struct fields at the END;
when a crash reproduces on the real workload but not in unit tests, check object and
executable mtimes against the changed header *before* reading the diff.

---

## 6. Operational notes

### 6.1 The two machines have SEPARATE checkouts
There is no shared filesystem. A design doc written on one box is invisible to an agent
on the other — this bit two handoffs. `scp` files across before referencing them, and
build **all four targets on both trees**:

```
mad-lab-main  build-hip  : llama llama-server llama-wp-expert-worker test-wp-expert-worker
mad-lab-2026  build-army : llama llama-server llama-wp-expert-worker test-wp-expert-worker   (-j2)
```

`-j2` on mad-lab-2026 is not optional: 15 GB RAM, i7-6700K, and it has been OOM-killed by
an over-parallel build before, taking out the desktop session.

### 6.2 mad-lab-2026 runs LIVE FLEET SERVICES from build-army
`pid 855466` (nemotron embedder, :8082) and `pid 3025042` (llama-router, :8093). Move the
active `libllama.so*` chain aside before rebuilding so they keep their mapped inode; never
signal or restart them. This worked cleanly tonight — both survived several rebuilds.

### 6.3 The backend landmine
`src/weight-pager/` has produced the **same bug class six times**: `#if defined(GGML_USE_*)`
that is compile-time satisfied but runtime-wrong. The precedent that matters for §1: the
RAM victim tier once had a raw `hipMemcpy` under a `GGML_USE_CUDA` guard satisfied on
Vulkan-only builds. The worker runs on **ROCm, CUDA and Vulkan** — any device↔host copy
must dispatch on the *runtime* backend. If you write `#if defined(GGML_USE_...)` in the
upload path, stop.

### 6.4 terra/codex handoffs
`gpt-5.6-terra` did good work tonight and pushed back correctly twice (refused a
speculative fix; flagged that `eval - ns_total` could not be attributed to barriers —
it was right and I was wrong). **It is currently blocked**: its environment has no SSH
agent (`SSH_AUTH_SOCK` empty on main), so it cannot reach the target. Fix that before
relying on handoffs, or expect to implement directly.

---

## 7. Harnesses

In `/home/kmbandy/.claude/jobs/87d16c2e/tmp/` **on mad-lab-2026** (they ssh to main):

| script | what it does |
|---|---|
| `stage7.sh` | 3 workers + spine, `WP_DISPATCH_STATS=1` + `WP_WORKER_STATS=1`, then kills the RX 480 to prove survival. **The main harness.** |
| `stage8.sh` | 2-worker variant (drops the 480) — changes experts-per-request, used to fit F and M |
| `stage9.sh` | stage8 with the R9700 worker wrapped in `gdb -batch -ex run -ex "bt 25"` — how the ABI crash was caught |
| `stage5.sh` | earlier 3-worker + kill test |

Run as `WSTATS=1 ./stage7.sh`. All have the port guard and SIGKILL escalation. Run-to-run
variance is **±3%** — do not claim anything below ~5% without repeats.

---

## 8. Open questions

1. **Why is H2D 1.29 GB/s?** Pinning, double copy, or synchronous-per-page. Unverified.
2. **Is GLM-5.2 routing predictable enough for prefetch?** Unmeasured. The DS4 numbers do
   not transfer automatically.
3. **The R9700's own F and M could not be separated** — it sat at ~5.35 experts/request in
   both configs (it holds 85–255 either way). The "3.5× slower" figure fits the 1070's
   line against the R9700's measured point.
4. **`sum = ggml_add(sum, result)` then `ggml_cpy(sum, result)`** in the two-phase batched
   path reads and writes `result` in one graph. It appears to work; nobody has confirmed
   the graph allocator cannot alias it.
5. **The two machines are on different commits** — main `0f1034aba`, 2026 `2e5c2c177`.
   Reconcile before any further merging.
6. **Host victim tier is built and default-off.** Worth ~8–10% at best (it avoids the
   NVMe read but still pays the same ~9.45 ms upload). Keep for prefetch staging later;
   not a priority.

---

## 9. Uncommitted work

All of tonight's changes are **uncommitted** on both machines. Modified:
`src/pipeline/pipe-expert-dispatch{er,-graph}.{h,cpp}`,
`tools/wp-expert-worker/{main.cpp,wp-expert-worker.cpp,wp-expert-worker.h}`,
`tests/test-wp-expert-{dispatcher,worker}.cpp`, plus new design docs in `docs/dev/`.
The tree is shared with another session's dirty files — **stage by explicit path only**,
never `git add -A`.

Contents: failure isolation, per-leg dispatch instrumentation, per-worker wait
attribution, worker phase timing, bulk pack/unpack, host victim tier (default off),
batched expert compute, and the broadcast fix.
