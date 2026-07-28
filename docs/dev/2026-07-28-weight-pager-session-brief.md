# Weight pager — 2026-07-28 session brief

Supersedes the ordering in `2026-07-27-prefetch-architecture-brief.md`. That brief's §5 plan is
now largely obsolete: item 1 is done, items 2/3/5 target problems we no longer believe we have.
Read §5 and §6 here first.

---

## 0. The short version

Four things changed what we believe:

1. **DFlash works, and it pays — but only with its confidence gate set.** At the `--spec-draft-p-min`
   default of `0.00` it costs 2.76x the expert page-ins and roughly halves throughput. At `0.3` the
   I/O amplification collapses to **1.11x** and throughput goes positive.
2. **Prefetch cannot reduce I/O, only move it.** A perfect oracle avoids 318,931 misses using
   317,892 prefetch loads — a ratio of 1.00. So no predictor, however good, reduces bytes.
3. **Eviction is the largest unclaimed headroom.** Plain LRU 28.56% miss vs Belady MIN 13.13% at
   our real capacity: better eviction alone is **54% of all available gain**, with no prediction,
   no prefetch and no extra bandwidth. Nobody was working on it.
4. **Four separate levers were sitting behind dead defaults.** Three had never been switched on in
   any run. Two of them paid; one was worthless; one is structurally inert for our model.

The instrument that made all of this decidable: **`page_ins` at temp 0 is deterministic** —
reproducible to ~3 counts in 73,000 across runs spanning hours and a 2-point system-load swing,
on a box whose wall clock drifts 16-23%.

---

## 1. Shipped

On `master`, mad-lab-main. Nothing pushed to origin.

| commit | what |
|---|---|
| `0917244a7` | dflash: gated decoder blocks (`attn_gate` + `enc.aux_norm`) |
| `d46f94607` | laguna: expose per-layer inputs so DFlash can draft against it |
| `ad1ae4c96` | wp-repack: expert-major repack tool + layer-range coverage guard |
| `50aebcd48` | wip: snapshot of other sessions' in-flight work (NOT reviewed, separately revertable) |

First two propagated to mad-lab-2026 as `274eef485` / `9580c5601`.

**Artifact:** `/home/kmbandy/models/Laguna-S-2.1/laguna-s-2.1-DFlash-Q8_0-swa.gguf` — Q8_0
(2126.77 -> 1129.96 MiB) plus the `sliding_window=512` + per-layer pattern the vendor export omits.
Original BF16 untouched.

### 1.1 What it took to make DFlash draft at all

Four distinct blockers, each hiding the next:

1. **The loader never asked for 7 tensors that were in the file.** `expected 76, got 69` reads
   backwards: 76 is what the *file* has, 69 is what the arch loader *requested*. Nothing was
   missing. `dflash` is a family parameterised by decoder block (`dflash.decoder_arch = "laguna"`,
   `gating = per-head`) and we implemented only the plain qwen3-style block — which matches the DS4
   speculator tensor-for-tensor, so the gated variant had never run. Added `attn_gate` (softplus,
   projected from the pre-attention state, `wo` deferred out of `build_attn`) and `enc.aux_norm`
   (per-aux-layer RMS norm *before* the `fc` fusion, confirmed against vLLM's `laguna_dflash`).
   Both optional, so the DS4 drafter still loads.
2. **The export omits `sliding_window`.** Not a subtlety: without it the draft KV cache is sized for
   the model's full 1,048,576-token context and dies allocating 24 GiB. With it, the same draft
   builds a 24 MiB iSWA cache.
3. **`--spec-type draft-dflash` is required.** Passing only `-md` loads the draft and drafts
   *nothing* ("no implementations specified for speculative decoding").
4. **The target must publish hidden states.** Only `gemma4` and `qwen35` set `res->t_layer_inp`;
   laguna didn't, so DFlash asserted the moment it was enabled. Also needed the post-loop entry:
   the array is `n_layer+1` because index `il` is the *input* to layer `il`, so index 48 on a
   48-layer model is the stack output. That also proves the GGUF's `target_layers` is correct as
   written and needs no re-export.

---

## 2. Measurements

Confidence differs per number. Quoted accordingly.

### 2.1 DFlash acceptance (established)

```
draft acceptance = 0.17109 (58 accepted / 339 generated), mean len = 1.84
acc per pos      = (0.522, 0.188, 0.101, 0.014, 0.014)
```

Position-1 52.2% with clean monotone decay. Note `common_speculative_print_stats` is called **only**
from the server — `llama-cli` never prints acceptance.

### 2.2 The `p_min` sweep (I/O established, throughput directional)

Every arm verified the engine reported the requested gate value in its own log before its numbers
were used.

| arm | page_ins/token | vs OFF | tok/s | vs OFF |
|---|---|---|---|---|
| OFF | 575.8 | 1.00x | 1.57 | — |
| `p_min=0.00` | 1591.7 | **2.76x** | 0.67 | −57.3% |
| **`p_min=0.30`** | 636.5 | **1.11x** | 1.90 | +21.0% |
| `p_min=0.50` | 1084.9 | 1.88x | 0.93 | −40.8% |
| `p_min=0.70` | 902.9 | 1.57x | 1.13 | −28.0% |

Replication of OFF vs P30, three interleaved pairs: **+29.9% / +24.5% / +0.5%**. Signs agree 3/3 so
the direction is real, but **the magnitude is not pinned** — OFF's own within-arm spread was 17.4%
(1.55/1.55/1.82), comparable to the effect, and pair 3 (lowest load) collapsed to +0.5%. The I/O
ratio in the same runs was **1.105x, deterministic** (OFF 73702/73701/73701, P30 81468/81469/81466).

The curve is **non-monotonic** (0.3 best, 0.5 worst, 0.7 middle) and that is **unexplained**.
Settling it needs per-arm acceptance stats, i.e. `llama-server`.

### 2.3 Strike gate (established)

`WP_HOST_PREFETCH_STRIKES` 2 -> 1, four interleaved arms:

| | strikes=2 | strikes=1 |
|---|---|---|
| `strike_held` | 5,507 | **0** |
| `enqueued` | 5,528 | 9,051 (+64%) |
| `read` | 4,526 | 7,544 (+67%) |
| `promotions` | 2,895 | 4,763 (+65%) |
| accuracy | 63.96% | 63.13% (**−0.8 pts**) |
| useful (promo/page_ins) | 1.42% | **2.34%** |
| `dropped` | 0 | **0** |

The gate was **pure throttle, not a quality filter** — the selection-effect worry was wrong. But
it's 65% of very little: prefetch still serves 2.3% of page-ins. The two strikes=2 arms agreed to
**0.05%**.

### 2.4 Hot-slot protection (established negative)

`WP_HOT_HIT_THRESHOLD` 0/1/2/4/8:

| threshold | page_ins | evictions | hot_skips |
|---|---|---|---|
| 0 | 81,468 | 81,078 | 0 |
| 1 | 81,467 | 81,078 | **19,806,296** |
| 2 | 81,468 | 81,078 | 6,433,480 |
| 4 | 81,465 | 81,078 | 1,103,781 |
| 8 | 81,468 | 81,078 | 34,980 |

Fires 19.8 million times, changes **nothing**. Cause: `hit_count_` is per-**slot** and resets on
recycle (`wp-pool.cpp:458,:480,:538`), so it measures reuse within the current residency episode —
approximately a restatement of LRU position rather than an independent frequency signal. It
protects what LRU would keep anyway. Binaries were md5-pinned before/after (a Codex build shared
the tree) and unchanged, so the arms are valid.

### 2.5 Offline simulation

`/home/kmbandy/wp_logs/accounting/analyze-token-axis{,-v2}.py` over
`routing_capture_laguna_ud4.bin` — 2,491 decode steps, 1,170,770 expert accesses, 100% of trace.

**Page granularity (load-bearing, was nearly got wrong):** one page = one tensor. Laguna stores
consolidated `ffn_{gate,up,down}_exps` and the catalog synthesizes one sub-page per (role, expert),
so **one expert activation = 3 pages**. The real 9000-slot VRAM pool is therefore **3000 lines**,
not 9000. Reading the 9000-*line* row corresponds to ~51 GB of VRAM that does not exist, and doing
so produced an inverted conclusion in round 1.

**Calibration anchor:** at ~3000 lines the sim gives 28.56% miss and 402.7 pages/token vs measured
~32% and ~452 on real hardware. Two independent quantities matching within ~15% — the simulation is
faithful. Keep this check on any future edit.

**The decomposition, at our real capacity (3000 lines = 17.2 GB):**

```
plain LRU 28.56%  ->  Belady MIN 13.13%  ->  drive-ceiling oracle 0.04%
                      ^^^^^^^^^^^^^^^^^
                      eviction alone = 54.1% of all available gain
```

At the 6 GB RAM tier (1047 lines): 55.51% -> 31.91%, 42.6% of gain.

**Prefetch moves bytes, it does not reduce them:** unlimited oracle avoids 318,931 misses using
317,892 loads — ratio **1.00**. And **one token of lead is enough**: T=1 through T=5 are identical
at realistic capacity; at small capacity longer horizons are pure waste (loads/avoided 5.16 at
N=1024, T=5). This **contradicts** the 07-27 brief §6, which argued for 3-5 tokens of lookahead.

**Bandwidth:** 90% of the oracle benefit needs ~1.22 GB/s sustained (independent arithmetic said
1.247 — agrees). 0.91 GB/s captures 69%; 1.25 GB/s captures 92%. All far below the demand path's
measured 2.27-3.21 GB/s burst.

**Token working set is exactly 470 lines** (min = max = mean), which is why the unlimited oracle
hits ~0% at any capacity >= 512 — a capacity tautology, not evidence of feasibility.

**Churn (M1):** next-token expert retention 33.14% overall (early 30.90 / mid 35.91 / late 32.57),
falling to 21.15% at distance 8.

**Popularity (M4):** top 5% of lines take 27.44% of accesses, top 10% 39.60%, top 25% 63.42%.
Smallest static pinned set covering 50/80/90% costs 1,875 / 5,033 / 6,892 lines. Static pinning is
**dominated** by plain LRU at 3140 lines (72.7% hit) — but the skew is *why* Belady wins, and
dismissing M4 on the static-pinning result alone was too hasty.

### 2.6 What the existing logs already knew

Read out of runs we already had, no GPU needed:

- **Host prefetch is 95.4% accurate and supplies 2.8% of page-ins** (`host_spec_promotions` 2113 /
  `host_prefetch_read` 2215; `page_ins` 79,897). The RAM tier is an **eviction victim cache**, not a
  prefetch cache (`host_tier_stores` 79,484 vs prefetch's 2,215).
- **`host_prefetch_dropped = 0`** in every arm, even at 7,544 reads. The single-worker QD1 transport
  has never been saturated and remains unfalsified as a bottleneck.
- **The xlayer path is ~0% effective**: `pred_pages` 1,553,760, `resident_skips` 1,406,393 (90.5%
  already resident), `submitted` 30,388, `hit` **40** (0.13%), `spec_evict_unused` 22,785. Same
  `RouterPredictor` as the host path, catastrophically worse — because it predicts the *next layer
  within a token* (the wrong-axis problem), while the host path predicts across tokens.

---

## 3. Retractions and corrections

Every load-bearing claim that got checked today either changed or died.

1. **"+40% throughput from DFlash"** — RETRACTED. Interleaved replication gave +21.5% / −3.4% /
   +1.3%, signs disagreeing.
2. **"Prefetch should be routed through the fast demand engine"** — WRONG. `wp-file-io.cpp` contains
   **zero** locks (verified); the engine is eval-thread-bound, and `wp-eval-cb.cpp:456-458` says so
   explicitly. A separate engine is justified *by thread safety*; only its buffered/QD1 slowness is
   not. The legitimate integration point is the mutex-guarded O_DIRECT worker pool.
3. **"Positions 4-5 each drag a full expert working set, so `n_max=3` will help"** — WRONG. 5->3
   saved only 7.3%: tail positions mostly re-select experts already fetched.
4. **"Speculative decoding halves throughput on paged MoE, turn it off"** — RETRACTED. That was
   measured entirely at `p_min=0.00`, i.e. a verdict on *ungated* speculation.
5. **"Expert-major repack gives ~24% effective pool"** — REVISED. With uniform slots it is ~**12%**,
   because triple sizes vary by layer (5.438 / 5.836 / 6.586 MiB as quant type shifts). Size classes
   recover the rest.
6. **The morning's "DFlash shows no throughput win"** — that comparison passed `-md` **without**
   `--spec-type`, so speculation never ran in either arm. It measured nothing.
7. **M4 dismissed as useless** — too hasty; the skew it measured is the mechanism behind the Belady
   gap.

---

## 4. Method

### 4.1 `page_ins` is the instrument, not tok/s

At temp 0 routing is deterministic, so `page_ins` reproduces to ~3 counts in 73,000 across runs
hours apart and under a 2-point load swing — while tok/s for an *identical* config ranged 1.49-1.82
(±17%). Judge all pager work on counters. tok/s needs interleaved replication and a sign test, and
even then licenses a direction, not a magnitude, when the control's spread is comparable to the
effect.

### 4.2 Gate verification — the NEVER FORGET rule (`d4b2ea95`)

Four levers this session sat behind gates at dead or permissive defaults. **A gate at a permissive
default means you measured the ungated system; a gate that rejects everything means you measured
nothing.** Both produce verdicts about a *technique* from a *misconfigured build*.

The operative clause, which is what would have caught all of them: **never infer a gate is live from
the existence of its flag — print its actual runtime value and verify it.** On 2026-07-27
`WP_HOST_PREFETCH_MIN_CONF` was set in the environment *and printed in the log* and was still dead,
because `predict()` at `wp-pager.cpp:843` omitted the argument.

Corollary learned the same day: "dead default" means **untested**, and the inference cuts both ways
— `strikes` and `p_min` paid, `hot_hit_threshold` was worthless.

### 4.3 Other guards worth keeping

- **Calibration anchor**: check a simulation against measured hardware on *two independent
  quantities* before believing its verdict. It caught the granularity error.
- **Binary md5-pinning** before and after any sweep that shares a build tree. Cost one command;
  converted a silent-invalidation risk into a verified negative.
- **Pre-register the interpretation**, including the ways a result can be informative-but-negative,
  and the known mechanistic caveats — so they are not invented afterwards.

---

## 5. Lever inventory

### 5.1 Dead — do not retry

| lever | why |
|---|---|
| `WP_HOT_HIT_THRESHOLD` (hot-slot protection) | fires 19.8M times, zero effect; counter is per-slot and resets on recycle |
| Static hot-set pinning | dominated by plain LRU at the same capacity |
| Draft-driven prefetch via `tid2eid` | structurally inert for Laguna — `ffn_gate_tid2eid` exists only for DeepSeek-4 (`src/models/deepseek4.cpp:182`), and `collect_tid2eid_pages_` early-returns on empty tables (`wp-pager.cpp:4166`) |
| `n_max` tuning for spec-decode I/O | 5->3 saved 7.3%; the cost is the waste ratio, not draft depth |
| Raising queue depth (`WP_P2P_QUEUE_DEPTH` etc.) | previously measured worse; concurrency was fixed a different way on 07-27 |
| Cross-layer (xlayer) speculative prefetch | 0.13% hit, 75% evicted unused, 90% of predictions already resident |

### 5.2 Settled — adopt

- **`WP_HOST_PREFETCH_STRIKES=1`** — +65% useful prefetch, −0.8 pts accuracy. Free.
- **`--spec-draft-p-min 0.3`** whenever a draft model runs against weight paging. **Never** the
  `0.00` default.

Both currently need an env var or flag, which is how they stayed dead. Promoting them to defaults is
a one-line change each plus a standing-config amendment.

### 5.3 Open, ranked

1. **Repack consumption** — the tool is built and verified (§6); the pager side is not started.
   Wins: request count ~690 -> ~230 per token, plus ~12% effective pool (uniform slots) or ~24%
   with size classes. **This is the gate-4 blast radius** — see §6.3.
2. **Eviction with per-page history** — the largest unclaimed number (15.4 points at our capacity,
   54% of available gain). Needs ghost entries for non-resident pages (ARC B1/B2, 2Q, LIRS). A data
   structure, not a knob. Mechanism now understood: reuse is frequency-driven (M4 skew) and LRU
   discards exactly that.
3. **Promote `strikes=1` / `p_min=0.3` to defaults.**
4. **Refine `p_min` in 0.2-0.4** and explain the non-monotonicity. Needs `llama-server` for
   per-arm acceptance stats.
5. **The 6900 XT's idle 10.8 GB.** Hot-expert pinning (~45-50% of accesses by M4's skew) vs
   whole-layer split (~15%). TB3 bandwidth is **not** the obstacle — the residual is 12 KB, so even
   47 layers x 2 crossings is 1.16 MB/token = 0.43 ms = 0.07% of a token. The real cost of
   hot-expert pinning is a cross-device partial-sum reduction per layer; whole layers avoid it.
   Note `8aebec54` (VRAM victim tier is dead on TB3) still stands and is a *different* mechanism —
   resident experts computed in place never cross TB3.
   **Re-measure after repack**, since ~12-24% more effective pool changes what is worth offloading.
6. **Cross-machine split** — now unblocked by the repack's sharded output rather than requiring new
   work. Justified by the second independent I/O path, not by capacity.
7. **RAM tier sizing** — is the 2.8% prefetch supply structural or undersized? Partly answered
   (structural: the gates throttled it), but the size effect is untested. Capped by 16 GB system RAM.
8. **`GpuTransport::init` signature mismatch** — the CPU-only `llama` target does not link:
   declared with 4 params (`wp-gpu-transport.h:39`), non-HIP fallback defines 3
   (`wp-gpu-transport.cpp:451`). Pre-existing, one-line fix, blocks CPU-only builds of anything
   linking `llama`.

---

## 6. The expert-major repack

### 6.1 Why

Expert weights are **role-major**: all `ffn_gate_exps`, then all `ffn_up_exps`, then all
`ffn_down_exps`. One expert's three tensors sit hundreds of MB apart, so each token issues 3
scattered ~1.9 MB reads per active expert instead of one contiguous ~5.7 MB read.

Two wins: request count ~690 -> ~230 per token, and **slot padding collapse** — three 2,580,480 B
slots carry a 5,701,632 B payload = **26.3% waste**.

### 6.2 What exists now (`ad1ae4c96`)

`tools/wp-repack/` + `tests/test-wp-repack.cpp`. Built by Codex gpt-5.6-terra, reviewed and fixed
here.

- **Model-agnostic by construction** — verified: zero tensor-name matching; it links `PageCatalog`
  and uses the same classification the pager uses. Groups by `(block_idx, expert_idx)` over pages
  with `is_expert && !is_consolidated`; member count is whatever the group has (a non-gated MoE has
  2); sizes may differ per layer; members may live in different GGUF shards.
- **Sharded output**, layer-aligned, a group never spans a shard. Per-layer default so any split
  point stays a runtime decision. Self-sufficient per-shard sidecar index — a machine holding only
  its own shards can validate them without the rest — plus a global manifest with SHA-256 identity
  hashes.
- **`--verify`** memcmps every payload byte against the source GGUF. A repack is a permutation of
  identical bytes, so this is an exact gate.
- **Verified on real data**: Laguna layers 1 and 46, 256 groups / 768 members each, exact byte
  comparison. Its per-expert totals (5,701,632 / 6,119,424 B) match an independent Python GGUF
  measurement **to the byte**.

**Defect found and fixed in review:** `--layer-ranges` silently dropped any layer the ranges did not
cover — a skip loop advanced past groups below the first range and nothing checked all groups were
consumed. Since the ranges normally describe a machine split, one mistyped digit would omit a whole
layer while reporting success, and `--verify` cannot catch it (it validates what the index claims,
not what the model contains). Now: coverage required, uncovered layers named, fails unless
`--allow-partial`, which warns loudly and lists every omission. Verified end-to-end against the real
binary — errors naming all 45 uncovered layers, writes nothing, exit code 1.

### 6.3 What consuming it requires (not started)

- `page_to_slot_` 3->1 is trivial (repeated values). **`slot_to_page_` 1->N is the structural
  change**: it holds a single `int`, and `on_pool_evict_` (`wp-pager.cpp:1594-1622`) uses it to
  invalidate exactly one page. A partially-invalidated group leaves members with
  `page_loaded_=true` pointing at recycled memory — **the exact gate-4 silent-wrong-weights class**.
- `ensure()` returning an interior pointer is easy: `slot_ptr_(slot) + intra_offset`. Precedent
  exists — `is_pinned` pages already return a non-slot `resident_ptr`.
- **Size classes are required, not polish** — uniform slots leave ~17% waste because triple sizes
  vary. `PoolAllocator` already has the machinery (`alloc_slot_size_class_`, `slot_class_`,
  `free_by_class_`, `use_size_classes` at `wp-pool.cpp:232`), defaulting to false.
- Watch `max_page_size` 2.46 -> 6.59 MiB: it also sizes the HostPrefetcher buffer
  (`wp-pager.cpp:1392`) and the ensure bounce buffers (`:2771`) — ~211 MB vs ~79 MB at 32 buffers,
  on a 16 GB box already committing 6 GB to the RAM tier.
- **Verification is unusually strong**: temp-0 output bit-identical, `io_gb_read` identical to the
  digit, `page_ins` falls ~3x. Three exact gates, no statistics. Plus a unit test forcing group
  eviction and asserting all members invalidate together.

---

## 7. Open questions

- Why is the `p_min` curve non-monotonic (0.3 best, 0.5 worst, 0.7 middle)? Hypothesis: with `-n`
  fixed, an over-aggressive gate shortens drafts until acceptance per step falls and more steps are
  needed. Unverified; needs per-arm acceptance stats from `llama-server`.
- Does the throughput benefit of `p_min=0.3` shrink as system load falls? Pair 3 (lowest load) gave
  +0.5% vs +29.9%/+24.5%. Three points cannot establish it; test with load as a swept variable
  rather than inferring from incidental drift.
- Is the RAM tier's 2.8% prefetch supply improvable by size, or is the victim-cache role the right
  one for it?
- Does Laguna's hot set generalise across prompts? All captures are from a small prompt set; the DS4
  cross-domain Jaccard was 15%.

---

## 8. Gotchas

- **`/tmp` on mad-lab-main is tmpfs, 7.8 GB — i.e. RAM.** A full ~73 GB repack must target `/home`
  (336 GB free), never `/tmp`. A 2.9 GB test write came straight out of the 16 GB budget.
- **`common_speculative_print_stats` is server-only.** `llama-cli` never prints acceptance.
- **`--spec-draft-p-min` defaults to `0.00` (disabled)** and `WP_HOST_PREFETCH_STRIKES` to 2.
- **The remote shell on mad-lab-main is fish.** Heredocs and `$(...)` break over `ssh`; write
  scripts locally and `scp` them.
- **`bash -lc` over ssh starts in `$HOME`**, not the repo — use absolute paths.
- **`find / -xdev` skips `/home`** (separate mount), and **nvm-installed binaries are absent from
  non-interactive ssh PATH** — this is why `pi` appeared "not installed" when it was.
- **Codex reads the repo `CLAUDE.md`** and will run `npx gitnexus analyze` if told the index is
  stale. That consumed 5.9 GB with a 16 GB heap cap on a 16 GB box. GitNexus's index is also stale
  and unreliable for this repo (it could not resolve `load_arch_tensors` and reported changes to
  files nobody touched).
- **Handoff status can read `in_progress` after completion** — Codex often cannot call
  `handoff_complete`. Check the session rollout mtime and tail, never the process.

---

## 9. Standing config

Amends `b5c91af4`.

```
WP_RESIDENT_DENSE=1  LLAMA_WP_TRANSPORT=p2p
WP_HOST_BUDGET_BYTES=6442450944          # 6 GB RAM tier, always on
WP_HOST_PREFETCH=1
WP_HOST_PREFETCH_STRIKES=1               # NEW: measured, free
--weight-paging --weight-paging-slots 10500
--weight-paging-resident-device ROCm1
--weight-paging-ffn-island-device ROCm1
-md <draft> -ngld 99 -devd ROCm1
--spec-type draft-dflash                 # REQUIRED, or nothing drafts
--spec-draft-n-max 5 --spec-draft-p-min 0.3   # NEW: never leave p_min at 0.00
```

Device mapping: **ROCm0 = R9700 32 GB = paging device** (PCIe); **ROCm1 = 6900 XT 16 GB = resident
device** (Thunderbolt 3, ~2.7 GB/s). At 10500 slots the R9700 sits at 85% and the 6900 XT at 37% —
the latter holds every role that currently exists for it (attention island, lm_head, shared experts,
FFN island, draft model) and cannot be filled further without new work (§5.3 item 5).

**Amendment to item 4 of `b5c91af4`:** "DFlash ON when a draft model exists" must be conditioned on
`--spec-draft-p-min > 0`. At the default it is a large pessimization on a paged model.

---

## 10. KG references

`5db7b8e0` dflash root cause · `39fe7bf6` SWA metadata · `93946fd6` what it took to draft ·
`ae2bd42b` page granularity · `2b58c7fe` prefetch moves bytes · `66baa4aa` two I/O engines ·
`8d33a975` prefetch starved by gates · `d324c639` fix eviction first · `639c7ce9` tid2eid inert ·
`13449ff4` strike gate · `f9c7a065` + `1d798065` + `6380ac29` + `898d9825` + `8e21ea4f` spec-decode
arc · `9f4959db` hot-threshold negative · `c25f8630` repack design · `d4b2ea95` NEVER FORGET
(fleet feedback, shared)
