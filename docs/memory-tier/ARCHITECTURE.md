# Tiered KV cache — architecture

How the paged + tiered + semantic-prefetch stack hangs together. This
doc is for the engineer modifying or extending the tier code; it is
not a "how do I configure the cache" doc (that's
[USER-GUIDE.md](USER-GUIDE.md)).

The full epic context is in [Jira MAD-126](#jira-references) but Jira
isn't where engineers read documentation while debugging. The
authoritative source for design decisions is this file plus the per-
decision ADRs under [`adr/`](adr/).

---

## The three-tier model

```
 ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
 │  Hot (VRAM) │ ───▶ │ Warm (host) │ ───▶ │ Cold (SSD)  │
 │             │ ◀─── │             │ ◀─── │             │
 └─────────────┘      └─────────────┘      └─────────────┘
   GPU buffers          host RAM             per-layer files at
   sized by             staging              ${ssd_path}/paged/
   --kv-tiered          buffers              instance-${ID}/L*.{k,v}.bin
   HOT%
```

A KV block — `block_size` tokens × `n_kv_heads` × `head_dim` of K and
V data per attention layer — lives in exactly one tier at any moment.
The tier is identified by the block's physical id range:

- IDs `[0, n_blocks_total)` → GPU pool (hot)
- IDs `[n_blocks_total, n_blocks_total + n_warm_blocks)` → CPU pool (warm)
- Cold tier is keyed by `(seq_id, logical_block_idx)` rather than by
  physical id — `KvtcStore` in `src/memory-tier/mt-kvtc-store.{h,cpp}`
  maps those into per-layer file offsets.

Movement between tiers is one-directional per call but fully bidirectional
in aggregate:

| From | To | Trigger | Mechanism |
|---|---|---|---|
| Hot | Warm | `evict_lru_to_warm()` from `ensure_blocks_for` | `ggml_backend_tensor_get` into `warm_k_/warm_v_` host buffers |
| Warm | Hot | `restore_block_from_warm()` from semantic restore or kernel demand-fault | `ggml_backend_tensor_set` from host buffers back into the GPU layer tensor |
| Warm | Cold | `spill_one_to_cold()` when warm pool fills | `KvtcStore::write` to SSD; warm slot freed |
| Cold | Hot | `restore_semantic_paged()` cold-fault path | `KvtcStore::read` → host buffer → `ggml_backend_tensor_set` |
| Cold | (drop) | Cold pool full + new spill needed | `drop_oldest_cold_block()`; **data is lost**, table entry becomes `kInvalidBlockId` |

Block contents are F16 (or whatever `--cache-type-k/v` is set to) on
hot; F16 in `warm_k_/v_` host RAM; **int4-with-scale** on cold. The
int4 cold compression is per-block: each block stores a single
`scale: f32` followed by `ceil(n_elts/2)` packed int4 nibbles. Cosine
similarity ≈ 0.99 against the unquantized baseline; this is acceptable
because cold blocks were already going to be re-attended-to in
combination with their original-precision neighbors. See
[`src/memory-tier/mt-quant.cpp`](../../src/memory-tier/mt-quant.cpp).

---

## The paged-attention block model

Adapted from vLLM's BlockTable / BlockPool design (Apache 2.0). The
key idea: tokens are not stored contiguously per sequence; they're
stored in **fixed-size physical blocks**, and a per-sequence
**logical→physical table** records which block holds which range.

```
seq 0 logical view                physical pool view
┌────────┬────────┬────────┐      ┌────────┐  block 7
│ tok0   │ tok16  │ tok32  │      ├────────┤
│ ...    │ ...    │ ...    │      │ tok0…  │  block 0
│ tok15  │ tok31  │ tok47  │      ├────────┤
└────────┴────────┴────────┘      │ tok16… │  block 4
   ↓        ↓        ↓            ├────────┤
table_[0] = [0, 4, 7]             │ ...    │
                                  └────────┘
```

Why this matters:

1. **Eviction is block-granular**, not byte-granular. The evict path
   moves a single block (~16 tokens × per-token K/V row size) at a
   time. That's a clean unit for `ggml_backend_tensor_get/set`.
2. **CoW is cheap**. `seq_cp` increments per-block refcounts in
   `BlockPool` rather than copying bytes. A branched agent workflow
   creating 10 conversation forks pays one block-table copy per fork
   and zero K/V copies until a fork actually writes new tokens. See
   ADR-A10.
3. **Holes are first-class**. A partial `seq_rm` can wipe one logical
   block in the middle of a sequence by setting that table entry to
   `kInvalidBlockId`. The paged-attn kernel reads `kInvalidBlockId` →
   returns `-INFINITY` for the wiped positions → softmax weights →
   zero contribution. No cache-side gymnastics required.

The block size is fixed at construction (default 16 tokens). The
paged-attn kernel
([`ggml/src/ggml-cuda/mt_pagedattn.cu`](../../ggml/src/ggml-cuda/mt_pagedattn.cu))
supports `(head_size, block_size) ∈ {(128, 16), (64, 16), (256, 16),
(128, 32)}` and K cache types `{F16, Q8_0, TURBO4_0}`.

---

## Why hybrid+paged is THE primary path

(See [adr/A1-hybrid-paged-primary.md](adr/A1-hybrid-paged-primary.md).)

User direction 2026-05-10: "hybrid+paged is THE primary path; pure-
attention is least concern." All new tier features land on
`llama_kv_cache_paged`. Pure-attention via `mt::llama_memory_tiered`
stays passthrough; the legacy `llama_kv_cache_tiered` /
`server_tiered_cache` paths got removed in MAD-127.

The `mt::llama_memory_tiered` wrapper survives in a thin form for two
specific responsibilities:

1. **bge-small embed model ownership.** The wrapper holds a single
   `EmbeddingModel` instance shared across server lifetime, and exposes
   `embed_text(string) → vector<float>` to whoever needs to compute a
   fingerprint or query.
2. **Recurrent-state backup on hybrid models.** The recurrent half of
   a hybrid model (Gated Delta Net, Mamba, etc.) uses
   `llama_memory_recurrent`, whose `clear()` loses everything. The
   tiered wrapper backs up the per-seq recurrent state into host RAM
   on `seq_rm` so it can be restored later.

Everything else — block management, eviction, persistence, semantic
prefetch — is in `llama_kv_cache_paged`.

---

## Class hierarchy

```
llama_memory_i  (interface)
└── llama_memory_hybrid                    (composition for hybrid models)
    ├── mem_attn → llama_kv_cache_paged    ◀── the active tier layer
    └── mem_recr → llama_memory_recurrent  (recurrent-only state)

mt::llama_memory_tiered                    (thin wrapper, optional)
└── inner_ → llama_memory_hybrid           (when wrapping a hybrid)
    or     → llama_kv_cache_paged          (when wrapping pure paged)
```

For hybrid models routed through the tier stack, the runtime stack is:

```
server-side dispatch
    ↓
llama_context::decode
    ↓
llama_memory_hybrid (split into attn vs recr ubatches)
    ├─→ mem_attn = llama_kv_cache_paged ──── paged-attn kernel dispatch
    └─→ mem_recr = llama_memory_recurrent ── recurrent kernel dispatch
```

The `mt::llama_memory_tiered` wrapper sits **outside** this stack when
present — it intercepts `seq_rm` etc. for the recurrent backup
behavior, then delegates to its inner cache.

The server-context helper `mt_get_paged_cache(llama_memory_i*)` peels
through three nestings (raw paged / hybrid / tiered+hybrid) to return
the underlying `llama_kv_cache_paged*`. See
[`tools/server/server-context.cpp`](../../tools/server/server-context.cpp)
near the `mt_get_paged_cache` definition.

---

## The single-threading contract

(See [adr/A4-single-threading.md](adr/A4-single-threading.md).)

`llama_kv_cache_paged` and its `BlockPool` / `BlockTable` are NOT
internally locked. The server's main loop is the single mutator; HTTP
worker threads communicate with it via `server_queue`'s task channel
and never touch the cache directly.

Concretely:

- Tier-counter `_total` accessors (e.g. `evict_h2w_total()`) return
  `uint64_t` from the cache. They're `volatile`-style monotonic counters
  that the `/metrics` HTTP handler reads from a different thread —
  this is the **only** cross-thread read on the cache. It's safe-ish
  on x86_64 / aarch64 for monotonic 64-bit counters (atomic loads are
  single instructions).
- Any future async path (semantic prefetch on a worker thread, bge-
  small embed batched off the critical path, etc.) must gate on a real
  concurrency design — not "by accident."
- Debug builds include a thread-id assertion (`check_thread_id_()` in
  `llama_kv_cache_paged`) that traps if anyone other than the registered
  main thread mutates the cache.

This contract is documented in
[`src/llama-kv-cache-paged.h`](../../src/llama-kv-cache-paged.h)
near the class declaration and in
[`src/memory-tier/mt-block-pool.h`](../../src/memory-tier/mt-block-pool.h).

---

## The persistence model

(See [adr/A5-persistence-explicit.md](adr/A5-persistence-explicit.md).)

The cache supports explicit save/restore via the server's `/slots/save`
and `/slots/restore` endpoints. Crash recovery is **not** automatic;
the model is "clean shutdown saves, restart restores."

State written by `state_write()`:
- Block table per seq (logical→physical mapping).
- Hot-tier K/V tensors (or skip + reprefill — caller-configurable).
- Warm-tier K/V buffers.
- Cold-tier index sidecar (CIDX v1 magic = `0x58444943`).
- BlockSemanticIndex fingerprints (PSFI v1 magic = `0x49465350`).

Format magic numbers and version bytes let future format changes be
detected and rejected (rather than silently corrupting state).

The cold-tier sidecar is the recovery hinge: per-layer K/V files
contain the actual block data, but without the sidecar's
`(seq, lblock) → file_offset` mapping the cache can't read them back.
Hard crashes that miss the sidecar write produce orphan files —
`scripts/army/cleanup-cold.sh` removes them.

---

## The multi-instance model

(See [adr/A6-multi-instance.md](adr/A6-multi-instance.md).)

Multiple `llama-server` processes can share one `--kv-tier-ssd-path`
without colliding because each writes cold-tier files under a
per-instance subdirectory:

```
${ssd_path}/paged/
├── instance-main-r9700/
│   ├── L0.k.bin
│   ├── L0.v.bin
│   ├── ...
│   ├── instance.lock           # flock'd while the process is alive
│   └── index.bin               # CIDX v1 sidecar
├── instance-main-6900xt/
│   └── ...
```

`--instance-id` defaults to the process PID but is typically set
explicitly by the boot script for stable cold-resume across restarts.

The `flock`-based lockfile prevents accidental double-start: a second
process trying to open the same instance subdir fails fast with a clear
error rather than silently corrupting the on-disk state.

---

## The semantic prefetch model

(See [adr/A2-fingerprint-at-prefill.md](adr/A2-fingerprint-at-prefill.md)
and [adr/A3-semantic-prefetch-only.md](adr/A3-semantic-prefetch-only.md).)

Two paths share the same `BlockSemanticIndex` storage in
`src/memory-tier/mt-semantic.{h,cpp}`:

### Write path: server-side prefill trigger

When the server processes a prompt, after the prefill batch lands
and the block table is fully populated, the server walks the new seq's
**complete** logical blocks (any block with `n` < `block_size` of
its tokens written is skipped) and:

1. Decodes the original tokens for that block back to text via
   `slot.prompt.tokens` plus `common_token_to_piece`.
2. Calls `mt::llama_memory_tiered::embed_text(text)` to get an
   L2-normalized 384-dim BGE-small embedding.
3. Calls `llama_kv_cache_paged::record_paged_block_fingerprint(seq, lblock, embedding, tier=Hot)`.

CPU cost is ~5ms × n_complete_blocks per prefill, off the GPU
critical path. Skipped when the block already has a fingerprint
(the `has_paged_fingerprint(seq, lblock)` short-circuit).

### Read path: server-side prefill query

After every prefill the server also computes an embedding of the
**most recent** complete block (the "query" for purposes of semantic
recall) and calls
`llama_kv_cache_paged::restore_semantic_paged(seq, query_embedding,
top_k, threshold)`. The cache scores its stored fingerprints against
the query, picks the top-K above threshold, and faults each matching
block back from warm/cold to hot before kernel dispatch.

### Why prefill-time and not eviction-time

Eviction in the paged cache is internal — the server doesn't see
eviction events. Fingerprints written at eviction time would be
strictly *behind* the data they describe (the data has already been
evicted) and the timing is hard to control. Writing at prefill time
makes the fingerprint write a clean, predictable, additive operation
attached to the server's ordinary task lifecycle.

### Why semantic doesn't drive eviction

(See [adr/A3-semantic-prefetch-only.md](adr/A3-semantic-prefetch-only.md).)

bge-small drives prefetch only, not eviction. Eviction stays on the
hybrid attention/recency/frequency policy in
`src/memory-tier/mt-eviction.{h,cpp}`. The reasons:
- Training-task mismatch: bge-small is trained for retrieval, not for
  inference cache predictiveness.
- Hot-path latency: every eviction decision would need an embed call.
- Doesn't fix the structural problem (hot-pool fragmentation under
  multi-seq load — that's the MAD-120 admission control's job).

---

## Kernel dispatch

The paged-attn kernel
([`ggml/src/ggml-cuda/mt_pagedattn.cu`](../../ggml/src/ggml-cuda/mt_pagedattn.cu))
takes the layer's K and V tensors plus the block_table tensor and
context-lens / q-lens (per-batch tensors maintained by
`llama_kv_cache_paged::prepare_batch_tensors`).

Dispatch table:

```
type_k:    F16 | Q8_0 | TURBO4_0
(head, block):
  (128, 16)
  (64,  16)
  (256, 16)
  (128, 32)
```

Aborts on unsupported tuples with a clear error. Hybrid models with
attention layers that fall outside this dispatch table cannot use
paged-attn until the kernel is extended.

`kInvalidBlockTableEntry` (matches `mt::kInvalidBlockId`) is handled
in the kernel: any (seq, position) whose physical block id is the
sentinel returns `-INFINITY` as its attention logit. After softmax
this contributes zero weight to the attention output — equivalent to
"that token doesn't exist." Used by partial seq_rm and by cold-drop.

---

## Eviction state machine

```
                          ┌────────────────────────────┐
                          │ ensure_blocks_for(seq, N)  │
                          └─────────┬──────────────────┘
                                    │
            ┌───────────────────────▼─────────────────────────┐
            │ pool_.alloc_gpu()                                │
            └─┬─────────────────────────────────────────────┬──┘
              │ ok                                          │ kInvalidBlockId
              │                                             │
              ▼                                             ▼
       ┌────────────┐               ┌──────────────────────────┐
       │ DONE       │               │ evict_lru_to_warm()       │
       └────────────┘               └──┬─────────────────────┬──┘
                                       │ ok                  │ false (warm full)
                                       ▼                     │
                                ┌──────────────┐             │
                                │ retry alloc  │             ▼
                                └──┬───────────┘   ┌────────────────────┐
                                   │ ok            │ spill_one_to_cold() │
                                   ▼               └──┬─────────────────┬┘
                            ┌────────────┐            │ ok              │ false (cold full)
                            │ DONE       │            ▼                 ▼
                            └────────────┘   ┌────────────────┐  ┌─────────────────────────┐
                                             │ retry alloc    │  │ drop_oldest_cold_block() │
                                             └────────────────┘  └──┬───────────────────────┘
                                                                    │
                                                                    ▼
                                                   ┌──────────────────────────┐
                                                   │ retry; on failure: false │
                                                   └──────────────────────────┘
```

The bottom escalation (`drop_oldest_cold_block` → data loss with
sentinel) is the **last resort**. The drop counter
(`paged_evict_cold_to_drop_total`) is the operator's signal to
re-tune sizing.

---

## Where things live

| File | What's in it |
|---|---|
| `src/llama-kv-cache-paged.h/.cpp` | The hot/warm-tier cache. Block table, eviction, semantic restore, state save/load, multi-instance lockfile. |
| `src/memory-tier/mt-block-pool.h/.cpp` | Physical block allocator (GPU + CPU pools, refcounting, watermark). |
| `src/memory-tier/mt-block-table.h/.cpp` | Per-seq logical→physical mapping. |
| `src/memory-tier/mt-semantic.h/.cpp` | `SemanticIndex` (chunk-level) + `BlockSemanticIndex` (per-block); save/load PSFI v1. |
| `src/memory-tier/mt-tiered.h/.cpp` | Thin `mt::llama_memory_tiered` wrapper: bge-small ownership, recurrent backup. |
| `src/memory-tier/mt-quant.h/.cpp` | int4 / int8 quant helpers including the per-block scaled int4 used for cold compression. |
| `src/memory-tier/mt-kvtc-store.h/.cpp` | Cold-tier file I/O. |
| `src/memory-tier/mt-eviction.h/.cpp` | `TokenMetadataStore` and the hybrid eviction policy. |
| `src/memory-tier/mt-mover-attn.h/.cpp` | Attention K/V mover for `mt::llama_memory_tiered`. |
| `src/memory-tier/mt-mover-recurrent.h/.cpp` | Recurrent state mover. |
| `src/memory-tier/mt-embed.h/.cpp` | `EmbeddingModel` wrapper around the bge-small gguf. |
| `src/memory-tier/mt-config.h/.cpp` | `TieredConfig` parsing. |
| `src/memory-tier/mt-capacity.h/.cpp` | `TierCapacityManager` for non-paged tiered (legacy). |
| `tools/server/server-context.cpp` | Server-side dispatch: prefill-time fingerprint write trigger, semantic restore call, /metrics extension, /slots tier residency, MAD-141 admission deadlock guard. |
| `ggml/src/ggml-cuda/mt_pagedattn.cu` | The paged-attn kernel + dispatch. |
| `tests/test-mt-*.cpp` | Unit tests for tier primitives. |
| `tests/test-paged-*.cpp` | Integration tests against `llama_kv_cache_paged`. |
| `tests/stress/stress-paged-multi-seq.py` | HTTP-driven stress driver. |
| `scripts/test/run-army-matrix.sh` | Per-device matrix runner. |
| `scripts/army/*.sh` | Boot scripts. |

---

## Jira references

- **Epic**: [MAD-126](https://mad-lab-ai.atlassian.net/browse/MAD-126)
  — production-quality paged + tiered + multi-seq for the agent army.
- **Children**: MAD-127 through MAD-138. See the Epic description for
  the story map.
- **ADRs**: extracted from MAD-126's "Architecture decisions" section
  into [`adr/`](adr/) — ten files A1-A10 covering the resolved design.
