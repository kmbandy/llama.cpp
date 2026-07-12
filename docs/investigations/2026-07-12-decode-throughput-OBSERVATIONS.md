# Decode throughput on the mad-lab fleet — observations and metrics

**Date:** 2026-07-12
**Status:** OBSERVATIONS ONLY. This document deliberately proposes **no cause and no fix.**
**Audience:** independent investigators (see "Your task" at the end).

---

## The problem, stated plainly

Under a real concurrent agent workload, token **generation** on our fleet is far slower than we
expect, and we do not know why. Prompt **processing** looks healthy. We want to understand the
generation path.

A 36-cell research run (36 concurrent LLM agents, each doing an agentic tool-calling loop) takes
**~50 minutes**. We believe it should be much faster. We do not know what the ceiling should be.

---

## The system

**Model:** `LFM2.5-8B-A1B` — 8.47B total params, **~1B active** (MoE). Hybrid architecture:
a minority of attention layers, the rest recurrent/conv. Q8_0, 9.0 GB on disk.
(Relevant: `llama_memory_hybrid` — see `src/llama-memory-hybrid.cpp`.)

**Hardware:**

| card | arch | VRAM | max power |
|---|---|---|---|
| Radeon AI PRO R9700 | RDNA4, gfx1201 | 32 GB | **300 W** |
| RX 6900 XT | RDNA2, gfx1030 | 16 GB | **255 W** |
| GTX 1070 | Pascal, CUDA | 8 GB | — |
| RX 480 | gfx803, Vulkan | 8 GB | — |

**Server (the R9700 scout — the card most of the data below comes from):**

```
llama-server --model LFM2.5-8B-A1B-Q8_0.gguf
  --device ROCm0 --n-gpu-layers 999 --no-mmap --no-warmup
  --ctx-size 1572864 --parallel 24          # => 65,536 tokens per slot
  --cache-type-k turbo4 --cache-type-v turbo4
  --flash-attn on
  --kv-tiered 75,25,0 --kv-tier-paged-blocks
  --kv-tier-semantic-index <LFM2.5-Embedding-350M-Q8_0.gguf>
  --cache-ram 1000 --ctx-checkpoints 4 --reasoning-budget -1
```

**Workload:** 18 concurrent agent conversations on this one card. Each is an agentic loop
(model turn -> tool call -> model turn ...). Contexts observed in the 3k-10k token range,
growing over the run. `turbo4` is a custom 4-bit KV quantization; `kv-tier-paged-blocks` is a
custom vLLM-style block-indexed paged KV cache. Both are ours, not upstream.

---

## MEASURED

Everything in this section was read off the running system. Nothing here is inferred.

### 1. Prompt processing is fast; generation is slow

```
prompt processing : 334.94 tokens/s        (also seen: 480 tok/s on the 1070)
generation        : 0.96 - 1.84 tokens/s   per stream
generation        : ~4.8 - 5.4 tokens/s    AGGREGATE across all 18 streams (tg_3s)
```

`tg_3s` is llama-server's own 3-second aggregate generation rate for the whole card.

### 2. Generation degrades as contexts deepen

Aggregate `tg_3s`, per minute, from the start of a run:

```
16:21   15.29 t/s   <- run starts, contexts shallow
16:22    6.34
16:23    4.49
16:24    5.57
16:25    4.87
16:26    5.90
16:27    5.00       <- settles ~5 t/s and stays there
```

Every run we have observed follows this shape: opens high, decays within 2-3 minutes, plateaus.

### 3. GPU "utilization" oscillates; **power tells a different story**

Sampled every 3s on the R9700 (300 W card) while 8-18 agents were actively running:

```
use= 95%  power= 153 W      <- bursts
use= 97%  power= 152 W
use= 98%  power= 152 W
use= 99%  power=  67 W      <- still "99% busy", power collapses
use= 97%  power=  83 W
use=100%  power=  73 W
use=  7%  power=  52 W      <- long stretches here
use=  7%  power=  52 W
use=  6%  power=  51 W
```

Observed range: **51 W to 154 W on a 300 W card.** It bursts to ~half the power budget, then falls
back. Time-averaged utilization is low. `GPU use %` and power draw frequently **disagree** — the
card reports 99-100% "busy" while drawing 67 W.

### 4. An idle, loaded model pegs the "busy" metric

With a model loaded and **zero requests in flight**:

```
6900XT, empty card (no model) :   0% use,  33 W
6900XT, model loaded, IDLE    :  99% use,  79 W, sclk boosted to 2530 MHz
```

**This reproduces with a plain server** — no tiered KV, no paged blocks, no semantic index,
`--ctx-size 65536 --parallel 1`. So it is not caused by our custom KV stack:

```
llama-server --model LFM2.5-8B-A1B-Q8_0.gguf --device ROCm1 --n-gpu-layers 999 \
  --no-mmap --no-warmup --ctx-size 65536 --parallel 1 --flash-attn on
=> idle, no requests: 99% "use", ~79 W
```

### 5. CPU

Two threads in the llama-server child sit at ~100% CPU. The rest are idle.

### 6. Host-side

Host RAM never became a constraint (peaked 12.3 / 15.5 GB; a 13.5 GB alarm never fired).
Per-scout host RSS ~3.0-3.1 GB, stable.

---

## DERIVED (arithmetic, not measurement — check it)

- **KV size:** ~4.75 KB/token. Derived from: 9.0 GB weights + hot KV (75% of 1,572,864 tokens)
  observed as 14.85 GB total VRAM. Please re-derive; this is arithmetic, not a reading.
- **Useful decode FLOPs:** ~1B active params x 2 FLOP x ~5 tok/s ~= **10 GFLOP/s**. The R9700's
  peak is on the order of 10^5 GFLOP/s. Sanity-check this yourself.

---

## RULED OUT (each by measurement, not argument)

Do not re-chase these. Each was a live hypothesis that we killed with data.

1. **The semantic-index embedder is not the bottleneck.** It runs a 362 MB embedder
   **synchronously on the server's single inference thread** (`server-context.cpp`, the
   `embed_text_batch` call inside `update_slots`), serialized behind one mutex and one
   `llama_context`. That is a real serialization point. We built a context pool
   (`--kv-tier-semantic-parallel N`) and A/B'd it on the same card under the same load:
   ```
   parallel=1  -> ~4.8 t/s aggregate (steady state)
   parallel=3  -> ~5.4 t/s aggregate (steady state)
   ```
   ~12%, inside the run-to-run oscillation. **Noise.** Not the wall.

2. **Host RAM / swap pressure.** Never approached the limit (see above).

3. **The custom KV stack is not what pins the idle "busy" metric.** The plain-server control in
   MEASURED #4 reproduces it with the tiering, paging, and semantic index all off.

4. **A trap we fell into, twice — do not repeat it.** Generation rate *opens high and decays*
   (MEASURED #2). Any measurement taken in the first ~2 minutes of a run, or on a shallow context,
   will look ~3x better than steady state. Two of our own conclusions today were wrong because we
   compared an early-run sample against a deep-run one. **Compare like-for-like or you will fool
   yourself.**

---

## Open questions we cannot answer

- What *should* aggregate generation throughput be for a 1B-active model on this card, at this
  concurrency and context depth? We have no reference number.
- Why do `GPU use %` and power draw disagree so sharply?
- What is the card actually doing during the long 7%-use / 52 W stretches?
- Does concurrency help at all? We have never measured generation at N=1 vs N=18 with context
  depth held constant.

---

## Your task

**Investigate the generation path and reach your own conclusions.**

We are deliberately **not** telling you what we think the cause is, and we are **not** proposing a
fix, because we do not want to anchor you. Prior experience on this codebase is unambiguous: when we
hand a reviewer our hypothesis, we get agreement with our hypothesis rather than an independent look.
Several of our confident theories today were wrong, and the ones that survived were killed by
measurement, not by argument.

**Rules of engagement:**

1. **Run things. Do not just read code.** Every real finding on this project in the last week came
   from executing something — a probe, a benchmark, a single-variable A/B — not from reading. Reading
   finds what is written down; executing finds what is true.
2. **Isolate one variable at a time.** This is the only technique that has reliably worked here.
3. **Distrust `GPU use %`.** See MEASURED #3. Prefer power, and prefer wall-clock throughput over
   any utilization metric.
4. **State evidence tier for every load-bearing claim:** `executed` (I ran it, here is the output) >
   `source` (file:line) > `inferred` > `assumed`. If a claim would change a recommendation and it is
   not `executed` or `source`, say so explicitly.
5. If you conclude the answer is "this is fine / expected," say that. A negative result is a result.

**Hardware access:** the **RX 6900 XT (ROCm1, 16 GB, 255 W)** is free — benchmark on it freely.
The **R9700 (ROCm0)** may be running a production workload; check
`curl -s http://127.0.0.1:8090/models` and `rocm-smi` before you touch it, and **ask before loading
anything onto it.** Never exceed ~95% VRAM on any card. One test process at a time. Stop on the
first OOM.

**Write your findings to exactly this path** (do not write any other file, do not overwrite anyone
else's):

```
/home/kmbandy/GitHub/llama.cpp/docs/investigations/2026-07-12-decode-throughput-findings-<YOURNAME>.md
```

where `<YOURNAME>` is the investigator name you were given (e.g. `fable`, `codex`). Another
investigator is working the same problem independently and must not see your conclusions.
