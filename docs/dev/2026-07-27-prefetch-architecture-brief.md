# Weight-pager prefetch — architecture brief, 2026-07-27 (evening)

**Supersedes `2026-07-27-prefetch-state.md`** (written this morning). That document
describes gate 4 as open — it is fixed — and frames the problem as a correctness
race in the VRAM speculation path. That framing is too narrow and its throughput
figures are retracted. Read this instead.

Every number below was read from a log in-session. Where a claim is unverified or
inside the noise, it says so.

---

## 0. The frame (kmbandy, and it is the right one)

> Think of an Amazon distribution warehouse. The assembly line is the main
> process and it must keep moving no matter what, because that's what we're
> bound by. Little scanners read the order as it passes — the scanner isn't
> bothering anything or making a whole copy of anything. The robots zoom around
> and grab stuff off the shelves and put it on the line. **It shouldn't
> interrupt / copy / change anything.**

| warehouse | here | status |
|---|---|---|
| assembly line | the eval/decode loop | **the bound** — must never wait |
| scanner reading a label in passing | read routing off tensors already on the GPU | ❌ we stop the line and copy |
| robots picking from shelves | `HostPrefetcher`, NVMe → RAM, own threads | ✅ correct already |
| shelves | expert weights on NVMe | ✅ |
| staging conveyor | the 6 GB RAM tier | ✅ |

Two violations, both now named precisely:

1. **The scanner stopped the line.** D2H + `hipStreamSynchronize` + scalar GEMV,
   inline in graph execution, per layer per token. Measured at **9.8% of run**.
   Fixed tonight (→ 0.6%) by moving it off-thread, but see §5 — that treated the
   symptom.
2. **The scanner copies the warehouse.** `RouterPredictor::set_router` keeps its
   own host-RAM duplicate of `ffn_gate_inp` (~151 MB over 48 layers) and streams
   ~91 GB through the CPU per run — despite `ffn_gate_inp` being a **dense tensor
   permanently resident in the 6900XT's VRAM**, and `cur` (`t->src[1]`) being the
   live device tensor. Both operands are already on the same GPU. **Not yet fixed
   — this is tomorrow's first code task.**

---

## 1. What shipped tonight

| commit (2026 / main) | what |
|---|---|
| `fdb441155` / `ea3af4891` | **gate 4**: `prefetch_pages_batch` handed itself the same slot twice; two reads DMA'd into one buffer and a page was silently mapped to another expert's weights. Verified by output identity, not throughput. |
| `f358614a2` / `1ae824969` | **HostTier speculative sub-tier** (`WP_HOST_SPEC_TIER=1`): a mispredicted prefetch can no longer evict a page the GPU actually used. |
| `7184e3cfe` / `97761f3cd` | print the `host_spec_*` counters (they were populated but invisible). |
| `4d282469e` / `1caddfff6` | **critical-path profiler**: wall vs thread-CPU for the inline block. |
| `3a19435f8` / `876229c47` | **async prediction worker** (`WP_HOST_PREFETCH_ASYNC=1`), default off. |

All default-off except the gate-4 fix. Suite 78 pass / 0 fail. Main's other
sessions' dirty files untouched throughout.

---

## 2. The measurement that matters

Laguna, 9000 slots, 6 GB RAM tier, 184,365 ms run:

```
block wall           18,375 ms   10.0% of run
block CPU on-path    18,088 ms    9.8%
blocked in sync         287 ms    0.2%
predict GEMV only    16,812 ms    9.1%
```

Reproduced to **0.3%** across two arms while tok/s and the blocked figure swung
freely. `CLOCK_THREAD_CPUTIME_ID` does not advance while blocked and does not
accrue time stolen by other processes, so the CPU figures are load-immune and the
wall figures are not.

Sanity: 16,812 ms / 29,034 calls = **0.58 ms per predict**; 786,432 MAC / 0.58 ms
= **2.7 GFLOP/s** — exactly a scalar single-threaded float reduction. Mechanism
confirmed, not just magnitude.

Async worker result:

```
on-path CPU : 18,202 -> 1,343 ms   (9.7% -> 0.6%)
predict CPU : 16,890 -> 17,921 ms  HELD — moved, not skipped
dropped 0 · enqueued 14,517 · max_depth 49/64
page_ins, host_tier_hits, host_prefetch_read, output: all identical
```

**Throughput payoff NOT established.** INLINE ran at host load 7.93, ASYNC at
10.29; the +18% wall delta is inside this box's noise band. Removal proven,
payoff unproven — both halves stand.

---

## 3. Retractions

- **"prefetch costs 16%"** — noise. See §4.
- **"the HostTier speculative sub-tier is worth +10%"** — same batch, same doubt.
- **"12,288 stream syncs are the cost"** — they are **0.2-1.0%**. Wrong.
- **Predicted the GEMV at 12-25%; it is 9.1%.** Over-predicted.
- The morning brief's **"prefetch is correct now and correctly measured it does
  not pay"** — that judged the VRAM path with the confidence gate structurally
  unreachable and 4 of 5 stack components absent. Not evidence about prefetch.

**Survives**, because it rests on counters that were bit-identical across a 20%
wall-clock swing: prefetch ON reads **1,757 fewer page-ins and 3.36 GB less**
NVMe traffic than OFF, three for three.

---

## 4. Measurement rules for this box (mneme-code `da055d88`)

**tok/s is not an instrument on mad-lab-main.** It is a live desktop;
`xdg-desktop-portal` alone holds ~59% of a core and load swings 9.2 → 12.5 inside
a 20-minute window. Interleaved replication, 3× per arm, prefetch OFF vs ON:

```
OFF: 1.76, 1.48, 1.41   spread 22.6%
ON : 1.34, 1.64, 1.55   spread 19.9%
paired deltas: +31.3%, -9.8%, -9.0%   -> SIGNS DISAGREE
```

**A control bracket is not sufficient validation.** The blocked design that
produced the retracted 16% *passed* its bracket — `BASE` 1.57 / `BASE2` 1.56,
agreeing to 0.6% — because load happened to match at the batch's start and end
while differing in the middle. A bracket samples the confound at two points and
is blind to drift that returns.

Use instead:
1. **Deterministic counters** (`page_ins`, `io_gb_read`) for I/O claims — exact.
2. **Thread CPU time** for critical-path/CPU claims — load-immune.
3. **Replicates per arm, interleaved**, and report the paired-delta signs. Signs
   disagree ⇒ no effect, full stop.

---

## 5. Tomorrow, in order

**1. Fix the DFlash draft GGUF — this is the critical path, not a speed feature.**
`laguna-s-2.1-DFlash-BF16.gguf` fails to load: `done_getting_tensors: wrong
number of tensors; expected 76, got 69`. It is the **lead-time source** for
prefetch (§6). Either re-export the speculator or find the loader change that
orphaned it.

**2. Scan in place (§0 violation 2).** Dequantize each router **once** into a
persistent f32 VRAM buffer on the resident device (3 MB × 48 ≈ **151 MB**, on the
6900XT sitting at 28%); run the prediction GEMV there on the existing `wp_stream`
against operands already in VRAM; async-D2H **1 KB of logits** instead of 12 KB of
residual; top-k + softmax over 256 values on the CPU is microseconds.

| | now | GPU-side |
|---|---|---|
| CPU per predict | 0.58 ms | ~0 |
| D2H per predict | 12 KB residual | 1 KB logits |
| CPU memory traffic | ~91 GB | ~30 MB |
| host copy of W | 151 MB RAM | none |

This **subsumes** the async worker and removes its one downside (the worker
streaming 91 GB alongside the eval thread, contending for memory bandwidth). Keep
from `3a19435f8`: the off-thread handoff, the shed-don't-block queue, the
`hp_async_*` counters, the `RouterPredictor` shared_mutex.

**3. Wake the draft-driven prefetch path.** `set_draft_window(int n_draft)`,
`draft_tid2eid`, and `draft_prefetch_{calls,pages_submitted,pages_resident,
queue_blocked,harvested}` are all **already built and have never executed**,
because their feed is the draft model from step 1.

**4. Instrument timeliness, not precision.** A pick that is correct but late is
worthless — the item wasn't on the line when it was needed. We have only ever
scored hit rate. Measure **arrived-before-needed**.

Note the ordering: step 2 alone would only make a too-short horizon cheaper.
Steps 1 and 3 are where the value is.

---

## 6. Layer lookahead vs token lookahead — the axis error

**`WP_PREFETCH_LOOKAHEAD_K` counts LAYERS within the current token, not TOKENS.**
K=2 predicts layers L+1/L+2 from layer L's residual: a horizon of 2/48ths of a
token, ~24 ms at laguna's 588 ms/token. **Even K=48 buys exactly one token**,
because a residual-based predictor cannot know what the next token is.

| axis | horizon | source | status |
|---|---|---|---|
| layer `K` | ≤ 1 token; offline recall decays 0.714 (k=1) → 0.569 (k=4) | current residual | the only knob ever swept |
| **token** | **3-5 tokens** | **DFlash draft** | built, zero, blocked on step 1 |

This explains the "prefetch never pays" record more economically than any tuning
hypothesis: every sweep measured **precision at a horizon too short for the pick
to arrive in time**. The 2026-07-10 note ("NO LEAD TIME … sub-10ms horizon") and
the 2026-07-19 rule ("K=1 IS NOT A TEST OF PREFETCH") both circled this but stayed
on the layer axis and prescribed K≥2 — still sub-token, still not lead time. The
axis was wrong, not the setting.

**Arithmetic supporting a 3-5 token target** (measured tonight):
- 1.70 t/s ⇒ **588 ms/token**.
- 115,820 page_ins / 256 tok = **452 pages/token**; 221.5 GB / 115,820 = **1.91 MB
  average per page-in** ⇒ ~864 MB/token ⇒ **~1.47 GB/s** against a drive measured
  at 3.88 GB/s — **~38% utilised**, so the spare bandwidth genuinely exists
  (consistent with 2026-07-19: concurrency binds, not bandwidth).
- 3 tokens of lead = **1.76 s**; at ~2 GB/s spare that stages ~1,800 pages ≈ 4
  tokens of experts.
- The 6 GiB RAM tier holds 6144/1.91 ≈ **3,220 pages ≈ 7 tokens** of working set.

So a 3-5 token horizon is matched to **both** the available bandwidth and the tier
size. kmbandy's "30-50% of experts by 3 tokens out" needs ~1,350 staged pages —
comfortably inside both.

---

## 7. Forward: Kimi-K3

K3 ships a **DSpark** draft. The draft-driven prefetch path (§5 step 3) is
**model-agnostic once something produces future tokens** — `set_draft_window` /
`draft_tid2eid` care about drafted token ids and their hidden states, not about
which speculator produced them. So the plan is to **adapt K3's DSpark draft onto
the existing DFlash prefetch mechanism** rather than building a second path.

That makes step 1 doubly load-bearing: fixing the DFlash feed for laguna is also
the integration point K3 will land on. Related: `llama.cpp` PR #25173 (DSpark) is
already on the open-levers list.

K3 runs near ~12% residency where a working prefetcher has far more to do than
laguna does, so the lead-time work matters more there, not less.

---

## 8. Standing config (mneme-code `b5c91af4`)

RAM tier **always on**; prefetch and eviction operate **together** on it; VRAM
**filled**; DFlash **on** when a draft exists. Any arm deviating is an *ablation*
and must be labelled as one.

Device map, verified in-session:
`ROCm0` = **R9700 32 GB** = paging device (PCIe `0000:42:00.0`)
`ROCm1` = **6900XT 16 GB** = resident device (Thunderbolt `0000:0b:00.0`)
That placement is correct — paging belongs on the PCIe card.

### Gotchas found tonight
- **`WP_FFN_ISLAND_DEVICE` is unreachable** from any common-args binary:
  `common_params` defaults the field to `"off"`, which is non-empty and beats the
  env. Use `--weight-paging-ffn-island-device ROCm1`. With the flag it places
  619,708,416 bytes.
- **`--draft-max` was removed**; the current spelling is `--spec-draft-n-max`.
- The **6900XT is still only 28% used** even with the island; it had just 591 MiB
  of `shexp` tensors to move. The 151 MB of router weights from §5 step 2 is a
  better use of it.
- `3400 slots` is a **K3 proxy** (~12% residency), not a laguna config. Label
  which question an arm answers.

---

## 9. mneme-code decisions written today

`b5c91af4` standing full-stack config · `e5003a95` HostTier spec sub-tier ·
`da055d88` measurement methodology · `f5f8a621` prefetch on the critical path ·
`36ba82bd` scan-in-place / warehouse frame · `24f12fa7` layer-vs-token lookahead
