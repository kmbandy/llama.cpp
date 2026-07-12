# Decode throughput — findings (investigator: `fable`)

**Date:** 2026-07-12
**Card used:** RX 6900 XT only (`--device ROCm1`, rocm-smi `GPU[0]`, 16 GB, 255 W).
**R9700 (ROCm0) was never touched.** It held the production `lfm25-8b-r9700-murmur` workload for the
whole session. Every number below is from the 6900 XT — i.e. from the **slower** of the two cards.

**Contention control:** one test server at a time; the harness kills its server between configs.
Verified before/between runs:

```
$ ps -eo pid,args | grep "[l]lama-server"     # only the 8090 daemon + the R9700 production child
$ rocm-smi --showmeminfo vram --showuse
GPU[0] : GPU use (%): 0
GPU[0] : VRAM Total Used Memory (B): 384258048     <- 6900XT clean, 0.38/16 GB
GPU[1] : GPU use (%): 89                            <- R9700, production, not mine
```
No OOM occurred except one deliberate probe (config E, below), which I stopped immediately.

---

## TL;DR

**Decode is not slow. The model is extremely fast. The custom KV stack is slow, and it is slow in a
way that stalls decode for every slot at once.**

Three separate, **multiplicative** costs sit between the stock server and the production preset. I
reproduced the fleet's exact numbers on the 6900 XT and then peeled the flags off one at a time:

| # | config (all: depth 5745 tok, 200 tok generated/stream, same card) | wall | per-stream tg | decode-phase aggregate | median power |
|---|---|---:|---:|---:|---:|
| C | **stock** f16 KV, `--parallel 18`, N=18 | **26.9 s** | **21.1 t/s** | **380 t/s** | 255 W |
| H | + `turbo4` KV (par 24, N=24) | 61.8 s | 11.6 t/s | 278 t/s | 254 W |
| G | + `--kv-tiered 75,25,0 --kv-tier-paged-blocks`, ctx 1.5 M | 191.2 s | 5.04 t/s | 121 t/s | 106 W |
| F | + `--kv-tier-semantic-index` = **full production preset** | **1023.0 s** | **1.49 t/s** | 36 t/s | **74 W** |

Config F **is** the fleet: it reproduces `0.96–1.84 t/s` per stream, `~4.7 t/s` end-to-end aggregate,
and the `67–74 W`-on-a-big-card power collapse, all of which the OBSERVATIONS doc reports as the
production symptom. F/C = **38x wall-clock**.

The single largest term is the one the OBSERVATIONS doc lists under **RULED OUT #1** — the semantic
index. It was ruled out on a flawed A/B (see "Where the previous ruling-out went wrong").

---

## The reference number you did not have

> *"What should aggregate generation throughput be for a 1B-active model on this card?"*

**`executed` — `llama-bench`, 6900 XT, stock f16 KV:**

```
| lfm2moe 8B.A1B Q8_0 | ROCm1 | pp512         | 6170.05 ± 154.46 t/s |
| lfm2moe 8B.A1B Q8_0 | ROCm1 | tg64          |  114.88 ±   0.41 t/s |
| lfm2moe 8B.A1B Q8_0 | ROCm1 | pp512 @ d2048 | 5889.51 ±   7.65 t/s |
| lfm2moe 8B.A1B Q8_0 | ROCm1 | tg64  @ d2048 |  115.47 ±   0.93 t/s |
| lfm2moe 8B.A1B Q8_0 | ROCm1 | pp512 @ d8192 | 5104.20 ±  73.34 t/s |
| lfm2moe 8B.A1B Q8_0 | ROCm1 | tg64  @ d8192 |  112.97 ±   1.10 t/s |
```

Three things fall out of this immediately:

1. **Single-stream generation is ~115 t/s**, on the *slower* card. Not 1–2 t/s.
2. **Generation does not decay with context depth** in the stock path — 115 / 115 / 113 at depth
   0 / 2048 / 8192. The decay described in MEASURED #2 is **not** a property of the model or the
   card. It is manufactured by the stack. (Mechanism below.)
3. **Prompt processing is ~5100–6170 t/s**, not 335. The OBSERVATIONS doc calls pp "healthy" at
   334 t/s. Measured against this reference, **pp is ~15–18x degraded too.** The problem was never
   generation-specific; both phases are degraded, and pp degradation is what starves decode.

**Batched decode also scales fine.** Open question #4 ("does concurrency help at all?") — yes, and
this is `executed`, depth held constant at 5745 tokens throughout:

```
stock, N=1  : 111.94 t/s/stream  ->  112 t/s aggregate
stock, N=8  :  35.27 t/s/stream  ->  282 t/s aggregate
stock, N=18 :  21.11 t/s/stream  ->  380 t/s aggregate
```

Concurrency buys **3.4x** aggregate decode. The batching machinery works.

> **Metric caveat, stated plainly because I got this wrong mid-run:** my harness's naive
> `tokens / wall` folds the prompt phase into the denominator and understates decode. The
> "decode-phase aggregate" column is `N x predicted_per_second`, which is the server's own timing of
> the generation phase only. Both are reported above so you can check me.

---

## What I ran

Harness: `llama-server` on ROCm1, one config at a time, killed between runs. Load generator fires N
concurrent `/v1/completions` requests with an **identical 5745-token prompt** (`cache_prompt: false`,
`temperature: 0`, `n_predict: 200`).

**Context depth is pinned at 5745 tokens in every single config.** This is deliberate — it is the
decay trap (RULED OUT #4). Every row in the ladder is like-for-like; none of the deltas below can be
explained by one sample being early-run and another deep-run, because they are all at the same depth
with the same generation count.

Configs, each differing from its neighbour by **one** group of flags:

- **A/B/C** stock: `--flash-attn on`, f16 KV, no custom flags. N = 1 / 8 / 18.
- **D** C + `--cache-type-k/v turbo4`.
- **E** C + `--ctx-size 1179648` (f16). **OOM — stopped, reported, not retried:**
  ```
  E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 13824.00 MiB on device 1: cudaMalloc failed: out of memory
  E llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache
  ```
  This is a useful negative: f16 KV at 1.18 M ctx wants 13.8 GB. **This is why `turbo4` exists.** Any
  recommendation to drop turbo4 must also shrink ctx.
- **F** the full production preset (`lfm25-8b-6900xt-murmur`, verbatim from `/models`).
- **G** = F **minus `--kv-tier-semantic-index`**, nothing else changed.
- **H** = turbo4 + jinja + cache-ram + swa-checkpoints, **no** tiering / paging / semantic.

---

## Finding 1 — the semantic index is the single biggest term, and it was not actually ruled out

**`executed`.** F → G removes *only* `--kv-tier-semantic-index`:

```
F  full production preset      wall 1023.0 s   per-stream 1.49 t/s   pp  15.4 t/s   power median  74 W
G  F minus semantic index      wall  191.2 s   per-stream 5.04 t/s   pp  48.8 t/s   power median 106 W
                               ----------------------------------------------------------------------
                               5.35x wall      3.4x tg               3.2x pp        +32 W
```

**`source` — the mechanism.** `tools/server/server-context.cpp:4136`:

```cpp
const auto embeddings = mt_tier->embed_text_batch(new_texts);
```

This is a **synchronous call to a 362 MB embedder, inside `update_slots()`**, on the server's single
inference thread. `update_slots()` is the one loop that advances *every* slot. It sweeps every
complete block of the slot's prompt (`n_complete_blocks = n_toks / bsize`, block size 16 →
5745 tokens = **359 blocks per slot**) and embeds all of them before returning.

**`executed` — it is strictly serialized and it freezes everyone else.** Server log from run F,
`--kv-tier-semantic-parallel 3` already set:

```
1:34.880  slot  6 | tier semantic: prefill fingerprint sweep — 359 new, 0 already-fingerprinted ...
2:17.756  slot  7 | ... 359 new ...
2:55.896  slot  8 | ... 359 new ...
3:37.699  slot  9 | ... 359 new ...
4:21.476  slot 10 | ... 359 new ...
5:00.474  slot 11 | ... 359 new ...
...
10:08.223 slot 18 | ... 359 new ...
```

**~40 seconds per slot, one after another, never overlapping.** 24 slots x ~40 s ≈ 16 minutes; the
whole run was 17:12. The run *was* the sweep.

And this is the part that makes it a **decode** bug rather than a prefill bug: because the sweep runs
inside `update_slots()`, the slots that have **already finished prefill and are mid-generation are
frozen for the entire 40 s** while slot *k* embeds. Their `predicted_ms` keeps accumulating wall time
while zero tokens come out. That is exactly why per-stream tg reads 1.49 t/s while the card sits at
74 W: the GPU is not idle and it is not decoding — it is running a 350 M embedder on tiny batches,
which is why "GPU use %" says 99% and the power meter says 74 W of 255 W. **The two metrics were
never in conflict; they were describing the embedder.**

### This also explains the decay (MEASURED #2), which the stock path does not have

Sweep cost is `O(n_complete_blocks)` = `O(context_length)` (`source`, line 4118). As agent contexts
grow from 3k → 10k tokens, each sweep gets proportionally longer, so the stall window per slot grows,
so aggregate tg falls — and then plateaus once contexts stop growing quickly. Stock `tg` is **flat**
with depth (115/115/113 above). The decay curve is a property of the sweep, not of attention.
`inferred`, but from an `executed` flat-baseline plus a `source` O(n) cost — I'd call it solid.

### Where the previous ruling-out went wrong

RULED OUT #1 varied `--kv-tier-semantic-parallel` between 1 and 3 and saw ~12%, and concluded
"noise, not the wall." That A/B varied the **size of the embedder context pool** — but the pool is not
the serialization point. The **call site** is: `embed_text_batch()` is invoked synchronously from
`update_slots()` and must return before any slot advances. Making the pool wider cannot help when the
caller blocks on the result anyway. The experiment that was never run is *removing the flag*, and
that is worth **5.35x**, not 12%.

---

## Finding 2 — the tiered/paged KV stack costs another ~3x

**`executed`.** G → H removes `--kv-tiered 75,25,0` + `--kv-tier-paged-blocks` and drops ctx from
1,572,864 to 147,456 (these are coupled — the huge ctx only fits *because* of tiering):

```
G  tiered + paged, ctx 1.5M   wall 191.2 s   per-stream  5.04 t/s   pp  48.8 t/s   power median 106 W
H  no tiering/paging, ctx 147k wall  61.8 s   per-stream 11.60 t/s   pp 153.8 t/s   power median 254 W
                               ----------------------------------------------------------------------
                               3.09x wall     2.3x tg                3.2x pp        +148 W
```

The power reading is the tell: **106 W → 254 W.** With tiering off, the card finally pulls its full
budget. With it on, the card is stalled on block gather/movement, not compute. I did **not** profile
inside the paged cache, so **whether this 3x is inherent to block-indexed paging or is a fixable bug
in it, I could not determine.** That claim is explicitly *not* made.

Also `source`, and possibly unintended — the server log for the production preset reports:

```
srv load_model: initializing, n_slots = 24, n_ctx_slot = 1572864, kv_unified = 'false'
```

`n_ctx_slot` is the **full 1,572,864**, not `1572864 / 24 = 65,536`. The tiered path hands every slot
the entire context ("total ctx=1572864 (model sees full)"). Whether that is intended, I can't say —
but it means per-slot bookkeeping is sized against a 1.5 M window for a workload whose contexts are
3k–10k.

---

## Finding 3 — turbo4 costs ~2x, and you cannot simply drop it

**`executed`, `llama-bench`, one variable (KV type), same card:**

```
                     f16 KV              turbo4 KV
tg64  @ d8192      112.97 t/s          73.45 t/s (± 33.46 — note the variance)
pp512 @ d8192     5104.20 t/s        1453.44 t/s
```

turbo4 costs ~1.5x on tg and **3.5x on pp** at depth. Under concurrency (C → D, N=18, same depth) it
costs ~1.8x on aggregate. It is a real cost, but it is the *smallest* of the three and it is **load-
bearing for VRAM**: config E proves f16 at 1.18 M ctx OOMs at 13.8 GB. turbo4 is only removable in
combination with a much smaller ctx.

---

## Finding 4 (negative result) — the idle "99% busy / 79 W" is a red herring

MEASURED #4 is real and I reproduced the shape of it, but **it costs nothing.** Stock N=1 on the
loaded server measured **111.94 t/s**, which is within noise of `llama-bench`'s **114.88 t/s** on the
identical card and model. If the idle spin were stealing throughput, the server would be slower than
the bench harness. It isn't. This is a ROCm busy-wait on an idle queue inflating a utilization
counter. **Do not spend time on it.** It is also *why* `GPU use %` is untrustworthy (rule 3) — the
counter measures "queue non-empty," not "SIMDs retiring work."

---

## What I would change, and expected magnitude

Ordered by (measured payoff) / (effort). All magnitudes are `executed` ratios from the ladder above,
at fixed 5745-token depth.

**1. Get `embed_text_batch()` off the `update_slots()` critical path. — expect ~5.4x wall.**
   The fix is not "make the embedder faster" and not "widen the pool" (already disproven). It is that
   a fingerprint sweep must never block the decode loop. Two options:
   - *Cheap and immediate:* drop `--kv-tier-semantic-index` from the agentic fleet presets. This is a
     one-line config change and it is worth **5.35x measured**. Do this today unless semantic KV
     retrieval is actively required for correctness on these runs — and note the fleet is currently
     paying 5.4x for an index whose *retrieval* benefit nobody has measured.
   - *Proper:* move the sweep to a background worker with its own `llama_context`, and let
     `update_slots()` enqueue-and-continue. Fingerprints are metadata; a block that isn't yet
     fingerprinted can simply be treated as un-indexed until the worker catches up. Nothing in the
     decode path needs the embedding to exist *now*.

**2. Question whether tiering/paging should be on for 3k–10k-token contexts at all. — expect ~3.1x.**
   The preset provisions a **1.5 M-token** context and then pays a paging/tiering tax on every step,
   for a workload whose contexts were observed at 3k–10k. That is a ~150x over-provision. With
   `--parallel 24` and, say, a 16k-per-slot budget (`--ctx-size 393216`) plus turbo4, the tiered/paged
   machinery is unnecessary. `assumed` — I did not verify that 393216 x turbo4 fits in 16 GB; verify
   before adopting. The 3.09x itself is `executed`.

**3. Leave turbo4 alone** unless you also shrink ctx. Smallest term (~2x), and removing it OOMs at
   current ctx (`executed`, config E).

**Combined expected effect.** Config C vs config F is **38x wall-clock** at identical depth and
identical generated-token count. Even taking only items 1+2 and keeping turbo4, the ladder says
H vs F = **16.5x** (1023 s → 61.8 s). Applied to the 36-cell research run: **~50 min → roughly
9–10 min from item 1 alone**, and **~3–5 min with items 1+2**. Those extrapolations are `inferred` —
they assume the 36-cell run is dominated by the same per-step costs I measured, which I did not verify
against the actual research harness.

---

## What I could NOT determine

- **Anything about the R9700.** Out of bounds by instruction. Every number here is 6900 XT. The R9700
  is the faster card, so the *absolute* fleet ceiling is higher than my numbers; the *ratios* should
  carry, but that is `assumed`.
- **Whether the tiered/paged 3.1x is inherent or a bug.** I measured the cost; I did not profile
  inside `llama-kv-cache-paged.cpp`. It could be a fixable gather, or it could be the honest price of
  block-indexed paging. Worth a follow-up.
- **Whether fingerprints survive across agentic turns.** `has_paged_fingerprint(slot.id, lb)`
  (`source`, line 4124) is keyed by **slot id + logical block**. I measured a single turn, where every
  block is new ("359 new, 0 already-fingerprinted"). In a real agentic loop the cost *should* become
  incremental (only new blocks embedded) — **but if a slot is ever reassigned to a different
  conversation, or logical blocks are renumbered by a context shift, that cache would be invalidated
  and the full sweep would recur every turn.** If that happens, the semantic cost is far worse in
  production than the 5.35x I measured. **This is the highest-value thing to check next**, and it is
  `inferred` — I did not test it.
- **The retrieval-quality benefit of the semantic index.** I measured only its cost. If it is buying
  real accuracy, the right fix is item 1's "proper" option, not deleting the flag.

---

## Raw config used for the reproduction (config F)

```
llama-server --model /home/kmbandy/models/LFM2.5-8B-A1B-Q8_0.gguf \
  --device ROCm1 --n-gpu-layers 999 --no-mmap --no-warmup \
  --ctx-size 1572864 --parallel 24 --flash-attn on \
  --jinja --reasoning-budget -1 --cache-ram 1000 --swa-checkpoints 4 \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  --kv-tiered 75,25,0 --kv-tier-paged-blocks \
  --kv-tier-semantic-index /home/kmbandy/models/LFM2.5-Embedding-350M-Q8_0.gguf \
  --kv-tier-semantic-parallel 3 \
  --kv-tier-ssd-path /home/kmbandy/llama/kv-cold/lfm25-8b
```
Load: 24 concurrent `/v1/completions`, 5745-token prompt, `n_predict=200`, `temperature=0`,
`cache_prompt=false`.
Result: **wall 1023.0 s, 1.49 t/s per stream, 4.69 t/s end-to-end aggregate, median power 74 W** —
i.e. the production symptom, on a clean card, with nothing else resident.
