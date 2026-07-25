# Decode throughput findings - codex

**Date:** 2026-07-12  
**Investigator:** codex  
**Hardware used:** RX 6900 XT only (`--device ROCm1`). I did not load anything on the R9700.

## Conclusion

The fleet's decode wall is not the LFM2.5 model, the 6900 XT, or concurrency in the ordinary llama.cpp decode path. It is the fork's paged-block KV path under concurrent, deep contexts. The effect gets worse with both configured context capacity and active context depth, and it reproduces the fleet's low-power signature.

The immediate action I recommend is to add:

```text
--no-kv-tier-paged-blocks
```

and run a steady-state fleet A/B. On the 6900 XT, at the fleet's total context capacity, N=18, and exact-history contexts of about 10k tokens, this one-option change raised aggregate wall decode from **10.448 tok/s to 140.263 tok/s (13.4x)**. The prompt-cache counters were matched (`cache_n` about 9,908 and `prompt_n=1`). Mean board power rose from 92.1 W to 172.4 W because the card was doing useful work.

This recommendation is **executed evidence**. It does not depend on GPU busy percentage.

I do not recommend a code change yet. Disabling paged blocks is a tested configuration mitigation. The long-term code work should first profile and repair the paged decode dispatch for multi-sequence LFM batches, then repeat the same fixed-depth tests.

## Safety and method

- I first checked `/models`, `rocm-smi`, and processes. The R9700 retained a workload, so I did not touch it. The 6900 XT became free before testing.
- Every server used `--device ROCm1 --n-gpu-layers 999 --no-mmap --no-warmup`.
- One server process ran at a time.
- I used named slots and identical prompts. For the final load-bearing comparisons I appended the warm-up generation to the next prompt, producing an exact cache continuation (`prompt_n=1`) rather than replaying a prompt suffix.
- I sampled GPU[0] board power (the physical 6900 XT) and used aggregate generated tokens divided by request wall time as the primary metric.
- The final attempted diagnostic restart encountered a near-full unknown 6900 XT allocation, so I stopped immediately and did not kill the unknown process.

## Results

### 1. Plain llama.cpp decode is fast, and concurrency helps

Configuration: no tiering, no paging, no semantic index; total context 294,912; 18 slots; about 4,507 prompt tokens per slot.

Single request command output:

```json
{
  "wall_s": 2.306,
  "prompt_n": 4502,
  "pred_n": 128,
  "timings": {
    "prompt_per_second": 3887.116,
    "predicted_per_second": 112.589
  }
}
```

Fixed-depth concurrency sweep output (wall time includes a small, matched 519-token cache replay in this initial sweep):

```text
N=1   aggregate=106.783 tok/s  power mean=187.2 W
N=2   aggregate=127.747 tok/s  power mean=160.3 W
N=4   aggregate=180.804 tok/s  power mean=178.0 W
N=8   aggregate=244.830 tok/s  power mean=207.1 W
N=12  aggregate=276.606 tok/s  power mean=206.1 W
N=18  aggregate=322.600 tok/s  power mean=218.7 W
```

**Claim [executed]:** concurrency itself is beneficial on this model and GPU. N=18 did not cause the ordinary decode path to collapse.

**Claim [executed]:** the plain path's N=18 per-stream rates were about 21-23 tok/s, versus the fleet observation of roughly 1 tok/s per stream.

### 2. Turbo4 has a cost, but it is not the wall

I changed only K/V cache types to `turbo4`, retaining the same 294,912 context allocation and leaving tiering/paging/semantic indexing off.

```text
plain,  N=18: 322.600 aggregate tok/s
turbo4, N=18: 212.669 aggregate tok/s
turbo4, N=1 :  77.649 wall tok/s (85.322 server decode tok/s)
```

**Claim [executed]:** turbo4 cost about 34% at N=18 in this test. That is material, but it does not explain a drop from hundreds of aggregate tok/s to about 5.

### 3. Tiering without paged blocks is not the wall

I added `--kv-tiered 75,25,0` and explicitly disabled its automatic paging with `--no-kv-tier-paged-blocks`. Everything else stayed matched.

```json
{"N":18,"wall_s":20.163,"agg_tok_s":228.536,
 "individual_min":14.451,"individual_max":16.686,
 "power_mean":223.9,"power_min":120.0,"power_max":259.0}
```

The turbo4-only control was 212.669 tok/s.

**Claim [executed, negative result]:** the 75/25 tier split without paging did not reduce throughput beyond noise in this test.

### 4. Paged blocks reproduce the low-throughput, low-power behavior

I then changed only paged blocks from off to on at total context 294,912.

```text
tiered, paged OFF, N=18: 228.536 tok/s, mean power 223.9 W
tiered, paged ON,  N=18:  61.535 tok/s, mean power 126.7 W
```

The paged N=1 result remained healthy:

```json
{"N":1,"wall_s":2.575,"agg_tok_s":99.398,
 "individual_min":104.497,"individual_max":104.497,
 "power_mean":178.4}
```

**Claim [executed]:** paged blocks introduce a concurrency-dependent regression. They cut N=18 throughput by 3.7x while leaving N=1 near 100 tok/s.

**Claim [executed]:** the regression has the same power signature as the fleet problem: wall throughput collapses together with mean power, even though transient samples still reach about 250 W.

### 5. Configured capacity amplifies the paged regression

With paging still on, I changed only total context from 294,912 to the fleet value 1,572,864. Active prompt depth remained about 4.5k.

```json
{"N":18,"wall_s":53.644,"agg_tok_s":21.475,
 "individual_min":5.624,"individual_max":11.377,
 "prompt_n":[516],"cache_n":[3991],
 "power_mean":104.3,"power_min":71.0,"power_max":254.0}
```

The matched smaller-capacity paged result was 61.535 tok/s.

**Claim [executed]:** configured paged-cache capacity is itself a throughput variable: increasing it 5.33x reduced N=18 throughput another 2.9x at the same active depth.

The server startup also printed:

```text
tiered KV (paged): total ctx=1572864 (model sees full)
initializing, n_slots = 18, n_ctx_slot = 1572864
```

In the plain control it printed `n_ctx_slot = 16384` for total context 294,912 and 18 slots. This is a semantic difference in the tiered configuration, not merely a storage implementation detail.

### 6. Active context depth reproduces the decay

At full fleet capacity with paged blocks on, I increased only active prompt depth from about 4.5k to about 10k.

An initial test resent the pre-generation prompt and therefore replayed 5,916 tokens. It produced 4.554 tok/s, but that number mixes prompt replay and decode and is not the primary decode result:

```json
{"N":18,"agg_tok_s":4.554,"cache_n":[3991],"prompt_n":[5916],
 "power_mean":114.8,"wall_s":252.981}
```

I corrected the test by appending the warm-up generation and continuing the exact slot history:

```json
{"N":18,"agg_tok_s":10.448,"cache_n":[9907],"prompt_n":[1],
 "individual_min":5.888,"individual_max":5.890,
 "power_mean":92.1,"power_min":78.0,"power_max":214.0,
 "wall_s":110.257}
```

**Claim [executed]:** the depth decay occurs under a fixed continuously active batch. It is not caused only by agent tool-call gaps or comparing an early run with a late run.

**Claim [executed]:** prompt replay can make the apparent wall rate even worse, but it is not required for the paged decode regression.

### 7. Disabling paging is a 13.4x mitigation at fleet scale

Final matched A/B, both with:

- model LFM2.5-8B-A1B-Q8_0
- total context 1,572,864
- parallel 18
- turbo4 K and V
- tier split 75/25/0
- exact-history context about 10k
- `cache_n` about 9,908 and `prompt_n=1`
- 64 generated tokens per stream

Only paged blocks changed:

```text
paged ON : wall 110.257 s, aggregate  10.448 tok/s, mean power  92.1 W
paged OFF: wall   8.213 s, aggregate 140.263 tok/s, mean power 172.4 W
ratio: 13.4x
```

Full non-paged output:

```json
{"N":18,"agg_tok_s":140.263,"cache_n":[9909],"prompt_n":[1],
 "individual_min":10.401,"individual_max":10.408,
 "power_mean":172.4,"power_min":78.0,"power_max":200.0,
 "wall_s":8.213}
```

**Claim [executed]:** `--no-kv-tier-paged-blocks` removes the dominant wall while retaining turbo4, tiering, full configured capacity, concurrency, and deep active histories.

## Source explanation and confidence

The measurements identify the component. Source provides a plausible mechanism, but I did not complete a profiler trace before the safety stop, so I label the detailed mechanism appropriately.

1. `--kv-tiered` automatically enables paged blocks unless explicitly overridden: `common/common.cpp:1638-1649` and the CLI description at `common/arg.cpp:1646-1654`. **[source]**

2. The LFM hybrid paged path uses `split_equal(..., sequential=true)` and feeds those ubatches to the paged cache: `src/llama-memory-hybrid.cpp:113-149`. **[source]**

3. For LFM2.5's 64-element head, the paged turbo path pads Q/K/V to 128 before the custom kernel and slices afterward: `src/llama-graph.cpp:2997-3017` and `3048-3054`. This imposes extra work but is not sufficient to explain the N=1 versus N=18 discontinuity. **[source plus inference]**

4. The custom flash-decode dispatch accepts only `total_q_tokens <= 8` and `num_queries_per_kv * total_q_tokens <= 16`: `ggml/src/ggml-cuda/mt_pagedattn.cu:1760-1799`. Otherwise it reaches the scalar fallback at lines 1840-1875. A normal N=18 decode step has up to 18 query tokens, so it is expected to miss this gate. **[source plus inference; not profiler-confirmed in this investigation]**

5. The scalar paged kernel walks the active context in 256-token chunks and loops token-by-token for QK and V accumulation: `ggml/src/ggml-cuda/mt_pagedattn.cu:1406-1512`. This predicts worsening time with context depth, matching the executed depth sweep. **[source plus executed correlation; causality not profiler-confirmed]**

6. The source itself records a prior regression investigation in which paged tile/decode/scalar paths collapsed relative to stock cache, at `ggml/src/ggml-cuda/mt_pagedattn.cu:1601-1606`. I did not treat that comment as evidence for my conclusion; I found it only after the A/B isolated paging. **[source]**

My confidence is **high** that paged blocks are the dominant fleet throughput wall and that disabling them is the correct immediate A/B. My confidence is **medium** that the decode-gate-to-scalar fallback is the specific internal mechanism. A `rocprof` trace or a successful `MAD_PAGEDATTN_PROBE=verbose` run should confirm that before changing kernel dispatch.

## Recommended next actions

1. On a non-production R9700 window, add only `--no-kv-tier-paged-blocks` and repeat the real workload through steady state (at least past the first 3 minutes). Compare aggregate generated tokens per wall second and board power. **[recommendation based on executed 13.4x A/B]**

2. If the non-paged configuration preserves the required tier/semantic behavior and VRAM margin, use it immediately. The flag is already the documented opt-out; no code patch is needed. **[source plus executed throughput; operational validation still required]**

3. For a long-term paged implementation, profile one exact-history N=1 and N=18 decode at 10k. Confirm the dispatch branch and kernel time before altering code. The first target is the `total_q_tokens`/GQA gate and scalar fallback, not the semantic embedder. **[inferred target, explicitly not yet executed]**

4. Add a repeatable benchmark covering N={1, 8, 18}, depth={1k, 4.5k, 10k}, fixed configured capacity, and both paging settings. Report aggregate wall decode and power after exact cache continuation. **[recommendation]**

## Arithmetic checks

- The observation document's KV estimate is unit-sensitive. Using decimal GB, `(14.85 - 9.0) GB / (0.75 * 1,572,864)` is about **4,959 bytes/token (4.84 KiB/token)**. Treating the displayed numbers as GiB gives about **5.20 KiB/token**. Therefore "about 4.75 KB/token" is the right order but is not a precise derivation without consistent units. **[derived]**
- `1B active parameters * 2 FLOP/parameter * 5 tok/s = 10 GFLOP/s` is arithmetically correct for weight operations only. It omits attention over context, routing, recurrent/conv work, quantization/dequantization, and memory traffic, so it is not a useful utilization estimate for this failure. **[derived plus inferred limitation]**

## Negative results retained

- Plain concurrency did not hurt; it improved aggregate throughput. **[executed]**
- Turbo4 alone did not explain the wall. **[executed]**
- Tiering without paged blocks did not explain the wall. **[executed]**
- Prompt replay worsened wall throughput but was not required to reproduce slow decode. **[executed]**
- I made no finding that GPU busy percentage is meaningful; all conclusions use wall time and power. **[executed method]**
