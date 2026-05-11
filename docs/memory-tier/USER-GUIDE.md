# Tiered KV cache — user guide

The tiered KV cache extends `llama-server` with a hot/warm/cold storage
hierarchy for the attention K/V state. It lets one server hold contexts
much larger than VRAM by spilling cold blocks to host RAM and SSD, with
optional semantic prefetch to bring the right blocks back when a query
revisits old material.

This guide is for someone running `llama-server`. For the architecture
behind it, see [ARCHITECTURE.md](ARCHITECTURE.md). For the army-style
multi-instance operations setup, see
[OPERATOR-RUNBOOK.md](OPERATOR-RUNBOOK.md).

---

## When to use it

Turn on the tiered cache when **any** of these is true:

- You want a context size larger than what fits in VRAM (e.g. 512k ctx
  on a 32 GB card).
- You're running multiple agents on one server (`--parallel 4`+) and
  want their contexts to coexist without OOM.
- You're running multiple `llama-server` instances on the same box and
  want each to fail fast rather than fight over VRAM.
- You expect long sessions where parts of the context get revisited
  (RAG-style queries, multi-turn agent workflows).

## When *not* to use it

Skip it when:

- Your context fits in VRAM with comfortable headroom and you only run
  one user.
- You're benchmarking raw decode throughput on a short prompt — the
  tier machinery adds bookkeeping that doesn't pay off until the cache
  actually fills.
- The model is pure-recurrent (no attention layers). The tier code is
  attention-block-keyed; recurrent state lives elsewhere and isn't
  paged.

---

## Quick start

The army-goal config — paged + tiered + semantic prefetch on a hybrid
model — is one extra line:

```bash
./llama-server \
  -m /path/to/model.gguf \
  --device CUDA0 -ngl 99 \
  --parallel 4 -c 524288 --no-mmap \
  --kv-tier-paged-blocks --kv-tiered 25,75,0 \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  --kv-tier-semantic-index /path/to/bge-small-en-v1.5-q8_0.gguf \
  --instance-id agent-1
```

Read it as: "give this server 524k context split 25% hot / 75% warm /
0% cold, with paged-attn block management and bge-small semantic
prefetch, identified as `agent-1` so its cold-tier files don't collide
with sibling instances."

For the same shape with a cold-tier slice on SSD:

```bash
... --kv-tiered 25,25,50 \
    --kv-tier-ssd-path /var/llama/cache \
    --kv-tier-cold-budget-mb 8192
```

---

## Flag reference

### Tier-shape flags

| Flag | Description |
|---|---|
| `--kv-tiered HOT,WARM,COLD` | Enable the tiered cache. Three percentages summing to 100. `25,75,0` = 25% VRAM, 75% host RAM, no SSD. `25,25,50` = SSD spillover. |
| `--kv-tier-paged-blocks` | Use vLLM-style block-indexed paged attention. Default-on for hybrid models when `--kv-tiered` is set; pass `--no-kv-tier-paged-blocks` to opt out. |
| `--kv-tier-paged-block-size N` | Tokens per paged block. Default 16; rarely needs changing. Must match the paged-attn kernel's supported block sizes (16 or 32). |
| `--kv-tier-total-ctx N` | Override the total ctx the tier sizing math uses. Defaults to `-c`. |

### Cold-tier (SSD) flags

| Flag | Description |
|---|---|
| `--kv-tier-ssd-path PATH` | Directory under which cold-tier files are written. Required when `COLD%` > 0. The server creates `paged/instance-${INSTANCE_ID}/` underneath. |
| `--kv-tier-cold-budget-mb N` | Cap the cold pool size to N MiB total (K+V across all attn layers). 0 = no cap (size from cold percentage). Use this to bound SSD wear. |
| `--kv-tier-cold-resume` | On startup, re-open the cold-tier files from the prior run instead of truncating. Pairs with `--instance-id`. Default: off (clean truncation). |
| `--no-kv-tier-cold-resume` | Explicit opt-out (overrides resume default if set elsewhere). |

### Multi-instance flags

| Flag | Description |
|---|---|
| `--instance-id ID` | Name this server instance. Used as the cold-tier subdir name (`paged/instance-${ID}/`) and to identify the lockfile. Default = the process pid. Required when running multiple instances against the same `--kv-tier-ssd-path`. |

### Semantic prefetch flags

| Flag | Description |
|---|---|
| `--kv-tier-semantic-index PATH` | Path to the embedding model (bge-small-en-v1.5 quantized to q8_0 is the tested config). When set, every block's contents are fingerprinted at prefill time; restore queries find semantically related blocks before kernel dispatch. |
| `--kv-tier-semantic-threshold F` | Cosine similarity threshold for semantic restore. Default 0.65. Higher = stricter match; lower = more aggressive prefetch. |
| `--kv-tier-semantic-topk N` | Maximum blocks restored per query. Default 5. |

### Compression flags

| Flag | Description |
|---|---|
| `--cache-type-k TYPE` / `--cache-type-v TYPE` | KV element type. `f16` (default), `q8_0`, or `turbo4`. `turbo4` is the int4-with-scale path used by the army stack and is ~4x smaller than f16 with cosine similarity ≈ 0.99 against the unquantized baseline. |

### Eviction policy flags

| Flag | Description |
|---|---|
| `--kv-tier-eviction-policy N` | Eviction policy id. The default Hybrid policy weights recency + attention attention magnitude + frequency. Reserved for tuning experiments. |
| `--kv-tier-attention-threshold F` | Lower bound on attention weight before a token is considered evictable. Default 0.10. |
| `--kv-tier-warm-device N` | Override the device id used to host warm-tier staging buffers. Default -1 (auto). |

### Deprecated spellings (still accepted, will warn)

| Old | New |
|---|---|
| `--kv-semantic-index` | `--kv-tier-semantic-index` |
| `--kv-semantic-threshold` | `--kv-tier-semantic-threshold` |
| `--kv-semantic-topk` | `--kv-tier-semantic-topk` |

---

## Sizing guide

The tier percentages multiply against `total_ctx × per_token_K_plus_V`
to size each pool. Pick from these starting points and tune from
metrics:

| GPU class | Suggested split | Rationale |
|---|---|---|
| 32 GB+ (R9700, 4090, A6000) | `25,75,0` | Hot pool generous; warm absorbs spillover; SSD only if you hit the warm ceiling. |
| 16 GB (6900XT, 4080) | `25,75,0` or `25,50,25` | Same shape; add cold if your context budget really exceeds VRAM+RAM. |
| 8 GB (1070, 3060, RX 480) | `25,25,50` | Hot is small; lean on warm + cold. SSD path required. |
| <8 GB | Don't run paged here for prod | Bookkeeping overhead starts to hurt. |

Rules of thumb:
- **Hot %** is what fits in VRAM with model weights still loaded. Don't
  exceed `(VRAM_total − model_weight_bytes − ~1 GB)`.
- **Warm %** wants host RAM headroom; on a 64 GB box budget no more
  than 32 GB for warm to leave room for the model loader and OS.
- **Cold %** needs writable SSD space at `--kv-tier-ssd-path`. Cap with
  `--kv-tier-cold-budget-mb` to bound wear.

The boot log prints the resulting allocation so you can sanity-check:

```
llama_kv_cache_paged: allocated 16/64 attn layers × 8.5 MiB (K+V) = 136.0 MiB total
                      (n_blocks=512, block_size=16, n_kv_heads=4, head_dim=256, type_k=turbo4, type_v=turbo4)
llama_kv_cache_paged: warm tier enabled — 384 host blocks × 17.0 KiB/block per layer × 16 attn layers = 102.0 MiB host warm storage
llama_kv_cache_paged: cold tier enabled — 256 blocks × 17.0 KiB/block (K+V) × 16 attn layers = 68.0 MiB on /var/llama/cache/paged/instance-agent-1
```

---

## Health metrics

Run with `--metrics` to enable the Prometheus endpoint at
`http://HOST:PORT/metrics`. The paged tier adds these keys:

| Counter | Meaning |
|---|---|
| `llamacpp:paged_evict_hot_to_warm_total` | Blocks moved hot→warm. Going up = warm tier is engaging (good when ctx exceeds hot, bad if you sized hot too small). |
| `llamacpp:paged_evict_warm_to_cold_total` | Blocks moved warm→cold. Going up = warm full, spilling to SSD. |
| `llamacpp:paged_evict_cold_to_drop_total` | Blocks dropped (no recovery). Going up = your cold pool is also full; data is being lost. **Tune sizing.** |
| `llamacpp:paged_restore_warm_to_hot_total` | Successful warm→hot fault-ins. Going up = workload is revisiting recently-evicted blocks. |
| `llamacpp:paged_restore_cold_to_hot_total` | Successful cold→hot fault-ins. SSD reads. |
| `llamacpp:paged_seq_preempt_total` / `paged_seq_restore_total` | Whole-sequence preemptions (MAD-120 admission control). |
| `llamacpp:paged_semantic_attempts_total` | Times the semantic restore path was invoked. |
| `llamacpp:paged_semantic_hits_total` | Attempts that restored ≥ 1 block. |
| `llamacpp:paged_semantic_blocks_restored_total` | Total blocks restored via semantic prefetch. Hit rate = `hits / attempts`. |
| Gauge | |
| `llamacpp:paged_blocks_capacity_gpu` / `_warm` / `_cold` | Pool sizes in blocks. |
| `llamacpp:paged_fingerprints` | Live BGE-small fingerprints currently held. |

The `/slots` endpoint extension shows per-seq tier residency
(hot/warm/cold block counts and live fingerprint count).

---

## Troubleshooting

### Server returns "paged KV admission could not fit a N-token request"

The submitted prompt is too large to coexist with the live workload on
the hot pool. Two ways out:
- Send a smaller prompt (chunk into multiple turns with `cache_prompt: true`).
- Wait for other slots to drain (decode-finish releases their blocks).
- Increase `-c` so more total blocks are available, **or** reduce
  `--parallel` so each slot has a larger share.

### `paged_evict_cold_to_drop_total` is climbing

Your cold pool is undersized for the workload. Either widen the cold
slice (raise `COLD%` in `--kv-tiered`), raise
`--kv-tier-cold-budget-mb`, or chunk requests so less old context lives
in the tier.

### `paged_semantic_attempts_total` is zero

Semantic prefetch fingerprints are written **at prefill time**, one per
complete (block-aligned) chunk of the prompt. If your prompts are
shorter than `block_size` (16 tokens), no fingerprints get written.
Either send longer prompts or increase prompt batch sizes so blocks
fill.

### `paged_semantic_hits_total / paged_semantic_attempts_total < 0.30`

The default 30% hit-rate target assumes cross-context queries.
Conversational workloads where every turn references only the most
recent N tokens won't benefit much from semantic prefetch — the LRU
path already keeps those hot. Lower
`--kv-tier-semantic-threshold` (e.g. 0.55) for more aggressive matches,
or accept that this workload doesn't need semantic prefetch.

### "instance lock /…/instance.lock held by another process"

Two `llama-server` instances tried to use the same `--instance-id` +
`--kv-tier-ssd-path` combination. Pick a unique `--instance-id` per
instance.

### Server aborts with `n_empty_consecutive > 3`

Pre-MAD-141 server. Pull the latest `feat/MAD-126-army-goal` branch
(or a release built after `0d66d8aa3`). The fix replaces the upstream
safety abort with a graceful per-slot `send_error()`.

### Cold-tier load reports "sidecar … missing or invalid"

The prior shutdown didn't write the cold-tier index sidecar — typical
on a crash or SIGKILL. The cold pool starts empty; existing per-layer
files are ignored on this run. Use the explicit `/slots/save` endpoint
before shutdown if you want the cold tier to survive a restart.
