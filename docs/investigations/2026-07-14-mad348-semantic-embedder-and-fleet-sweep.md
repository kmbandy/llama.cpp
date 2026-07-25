# MAD-348 — semantic-index cost, granite swap, and the 4-GPU fleet sweep

**2026-07-14, claude__main + kmbandy. End-of-day writeup. Nothing is committed yet.**

This was a long single-session arc that started from the parallel-decode synth-fleet brief and
ended up (a) fixing the semantic-index cost that was silently crippling long-context throughput,
(b) swapping the embedder model, (c) sweeping the fleet, and (d) surfacing ONE unexplained thing
worth chasing first thing tomorrow.

---

## TL;DR — the tomorrow headline

**Decode at depth does not scale with the card, and nobody knows why yet.**

| | 6900 XT | GTX 1070 | ratio |
|---|---|---|---|
| decode, SHORT ctx (synth) | ~490 tok/s | ~115 tok/s | 4.3x (as expected) |
| decode, DEEP ctx (6k, murmur) | **35.3** tok/s agg (par24) | **33.7** tok/s agg (par8) | **~1.0x (!!)** |
| prefill, deep ctx | 940 tok/s | 624 tok/s | 1.5x (fine) |

The 6900 XT has ~2x the memory bandwidth and far more compute than the 1070, and on short-context
decode it delivers 4.3x. **At depth that entire advantage evaporates** — the two cards tie on
aggregate decode, and per-slot the 1070 is actually *faster* (4.21 vs 1.47 tok/s/slot). Prefill
still scales with the card (1.5x), so this is specific to the **decode-at-depth path**.

Caveat to resolve tomorrow: the comparison is at different `--parallel` (24 vs 8) and different
quant (Q8 vs Q6), so it is not yet a clean single-variable result. But the *shape* is real: the
6900 XT decode collapses 490→35 (14x) going short→deep, while the 1070 only drops 115→34 (3.4x).

**This is the thing to look at first tomorrow.** It is almost certainly in our KV stack (paged
attention decode / tiering / semantic restore at depth on the ROCm/HIP path), not the GPU.
Note: a pure-decode test earlier this session showed the 6900 XT scaling FINE to par4 at 16k
(91→119 agg). So the collapse happens somewhere between par4 and par24 at depth — that bracket is
the experiment.

---

## What got fixed (MAD-348) — all in the llama.cpp working tree, UNCOMMITTED

The semantic index was costing ~2.4x at short context and ~3.5x prefill at 18k, while — this is
the kicker — **restoring nothing**. Five separate things, each measured:

### 1. Restore threshold was wrong for the model (the whole feature was dead)
`kv_semantic_threshold` default was **0.65**. That is BGE's number. The fleet swapped the embedder
to LFM2.5-Embedding-350M months ago, whose true-positive cosines top out at ~0.63 — so **0.65
rejected 100% of genuinely-correct matches** and the index prefetched nothing while paying full
cost. Measured per model (12 passage/query pairs, 144 pairings):
- LFM2.5-350M: TP min 0.332 / median 0.521 / max 0.634 → threshold **0.35**
- bge-small:   TP median 0.764 → threshold **0.63** (this is why 0.65 was once right)
- granite:     TP median 0.885 → threshold **0.84**
Fix: default → 0.84 (granite is now the fleet embedder). Regression test pins it in
`tests/test-paged-semantic.cpp`.

### 2. Embedder ran on 4 of 24 threads
`mt-embed.cpp` never set `cparams.n_threads`, so it inherited `GGML_DEFAULT_N_THREADS = 4` (whose
own definition is commented "TODO: better default"). On the 12-core box a 350M model ran on 4
threads: ~360 tok/s of embedding, 50-68s for an 18k-token sweep (phase probe: 100% inside
`embed_batch`). Fix: `n_threads` plumbed from `--threads` (per-box `common_cpu_get_num_math()`,
overridable via new `--kv-tier-semantic-threads`). **Deliberately NOT a hardcoded "cores-1"** —
that would give 11 on the 12c box but only 3 on the 4c/8t box (a REGRESSION on the small box).
Measured on 6900 XT: prefill 260→~400 tok/s (~1.6x). On the 4c box the lever is exhausted (4
threads either way).

### 3. Granularity was welded to KV paging block size
The fingerprint sweep embedded EVERY paged block (block size 16), so it was O(context / 16) embeds
— ~1128 for an 18k prompt. New `--kv-tier-semantic-span` (default 16 = 256-token passages)
decouples semantic granularity from paging granularity. Worth ~1.36x AND it raises cosine scores
(0.36 → 0.56) which is what makes the threshold robust. NOTE: it does NOT give the 16x a naive read
predicts — embedder cost is O(tokens) not O(calls), and both embed the same ~18k tokens.

### 4. Async sweep — BUILT then REMOVED
I moved the sweep off the inference thread with `std::async`. It core-dumped at depth+concurrency:
`llama-kv-cache-paged.h` documents a THREADING CONTRACT ("cache is SINGLE-THREADED … BlockPool /
BlockTable / BlockSemanticIndex have NO internal locking") and literally names "bge-small embedding
on a CPU worker parallel to GPU compute" as violation example #1 — exactly what I built. Removed
it. With span + threads the sweep is cheap enough to stay inline.

### 5. Embedder swap: LFM2.5-Embedding-350M → granite-embedding-small-r2 (47M, 7x smaller)
Granite was dropped in July because it SIGABRT'd — but that was OUR clamp bug (clamped chunks to
`llama_n_ctx` not `EMBED_BATCH_TOKENS`), since fixed. The 350M model was adopted to dodge a bug
that no longer exists. Measured: granite retrieves as well or better (AUC 0.975 vs 0.895 on
same-topic passages — top-1 is the wrong metric; the index thresholds+top-k, so SEPARATION is what
matters), 7x smaller, no VRAM, and **prefill at 18k: 468 → 1085 tok/s (semantic index now ~95% of
the no-index ceiling)**.

**The real deliverable is the COUPLING, not granite** (kmbandy will train an mlambaformer embedder
later). model + pooling + prefixes + threshold are ONE unit; stranding any one is SILENT. Now:
pooling reads from the GGUF (`LLAMA_POOLING_TYPE_UNSPECIFIED`, was hardcoded CLS); prefixes are
`--kv-tier-semantic-{query,doc}-prefix` (default empty); threshold tracks the default embedder; and
a startup line prints the whole unit: `granite…gguf | pooling=2 (from GGUF) | query_prefix=(none) |
doc_prefix=(none) | threads=N`.

### Bonus fix (proven): RX 480 semantic index RESTORED
The 480 lost its semantic index in July when the embedder's stray CUDA context OOM'd the full 1070.
The CPU-only device-list pin (part of this session's mt-embed.cpp changes — `n_gpu_layers=0` is NOT
enough, you must set `mparams.devices`) removes it. TESTED on 2026: both 1070+480 scouts loaded at
once, 480 served a token, ZERO CUDA OOM on the 1070.

---

## The "anti-scaling" red herring (refuted by profiling — do not chase again)

Mid-session I claimed paged decode "anti-scales" with concurrency at depth and nearly cut a kernel
for it. **rocprofv3 --kernel-trace refuted it**: aggregate decode SCALED 22.6→40.2 tok/s (conc
1→4), per-token GPU work IMPROVED. The apparent anti-scaling was MY test firing 4 fresh 16k
prefills that contended through the single loop — a prefill-contention artifact, not a decode
property. (This is distinct from the tomorrow-headline above, which is decode staying flat vs the
CARD, measured with prefill isolated.)

Minor real finding, low priority: `mt_pagedattn.cu:1819` sizes the split-K partials buffer with
`max_q_len = total_q_tokens` (SUM) when the correct stride is the MAX → O(num_seqs²) buffer. Real
but small (34MB conc4 / 140MB conc8 at 16k) and the profile shows decode attn is only 9-20% of GPU.
Not the lever. Fixing it needs host-side max(q_lens) per batch (graph reuse blocks static
op_params) and wrong stride = silent OOB corruption → brick-risk, kmbandy's kernel.

---

## Fleet sweep results

### Synth (short ctx, high fan-out) — peak decode throughput, per GPU
| GPU | knee `--parallel` | tok/s | ceiling |
|---|---|---|---|
| 6900 XT | 160 | ~490 (drops past 160: 386@192, 356@224) | VRAM |
| R9700 | ~96 | **~600, dead flat** across par 64-196 AND 1-2 instances, at 135-180W of 300W, 100% "use" | **memory bandwidth** |
| GTX 1070 | 24-32 | ~115 (flattened) | bandwidth/VRAM |
| RX 480 | **8** | ~80 (par>8 HURTS: 59@12, 50@16) | compute |

**R9700 bandwidth finding** (kmbandy: "it should cook"): on short decode it CAN'T, because decode
is memory-bandwidth-bound (reads ~1GB active MoE weights/token; 600 tok/s ≈ 600 GB/s ≈ its bus
ceiling). Two instances gave 599 vs single 558 — refuted host-starvation. The card cooks on
COMPUTE-bound work (prefill, deep ctx, calibration), NOT decode throughput. This is the "two
regimes" truth: inference decode = bandwidth-bound; prefill/calibration = compute-bound.

**3-GPU simultaneous synth aggregate: 645 tok/s** (6900XT 447 + 1070 110 + 480 89). GPUs don't
contend. Projected fleet with R9700 (~600): **~1245 tok/s**.

### Murmur (deep ctx, production config: granite + tiering) — prefill vs decode SEPARATED
| GPU | par | PREFILL agg | DECODE agg |
|---|---|---|---|
| 6900 XT | 24 | 940 tok/s | 35.3 tok/s |
| GTX 1070 | 8 | 624 tok/s | 33.7 tok/s |
| RX 480 | 8 | (not captured — run timed out) | (not captured) |

→ this is the table that produced the tomorrow headline. Prefill scales with the card; decode at
depth does not.

---

## State of the tree (READ THIS before committing)

**NOTHING is git-committed on either box.** All working-tree changes.

- **mad-lab-main** (HEAD 6ce3be4d): the 15 MAD-348 code files changed + `router-fleet-main.ini`
  (production murmur scouts swapped to granite/0.84) + this doc + RESULTS.md.
  Session-start pre-existing WIP on common/{arg,common.h,common.cpp}, include/llama.h,
  llama-context.cpp, llama-batch.cpp, tools/server/server-*, and the DSWS spike dir — NOT mine,
  leave alone.
- **mad-lab-2026** (HEAD fee38ef7e, build-army = CUDA+Vulkan): same 15 files copied over (verified
  the only delta was my changes; router WIP preserved), rebuilt clean, all 5 tiered tests pass.
  `router-fleet-2026.ini`: scouts + RX480 swapped to granite/0.84. Rollback copies of 2026's
  original files in scratchpad/from2026/.

### The 15 MAD-348 code files
common/{arg.cpp, common.cpp, common.h}, include/llama.h, src/llama-context.cpp,
src/llama-kv-cache-paged.h, src/llama-memory.h, src/llama-model.cpp,
src/memory-tier/{mt-config.h, mt-embed.cpp, mt-embed.h, mt-tiered.cpp, mt-tiered.h},
tools/server/server-context.cpp, tests/test-paged-semantic.cpp.

### CLEANUP OWED before commit
1. **Ablation aliases litter both presets** — main: `sw-6900xt-synth*`, `sw-r9700-synth*`,
   `lfm25-8b-6900xt-{synth,synth64,synth128,ablate-nosem,paged-warm,paged-nowarm,plainkv,pagedonly,
   evict,semt2..16,span1,murmur-sp8}`, `bge-small-embed`, `granite-embed`, `murmur-emb-*`,
   `lfm25-embed-350m`. 2026: `sw-1070-synth*`, `sw-480-synth*`. Strip all before commit.
2. RESULTS.md (`2026-07-14-parallel-decode-synth-fleet-RESULTS.md`) has a STALE anti-scaling
   section — that finding was refuted. Correct or delete.
3. Review pass on the ~15-file diff.

---

## Tomorrow, in priority order

1. **DECODE AT DEPTH ON THE 6900 XT** (the headline). Clean single-variable experiment: 6900 XT
   decode tok/s as f(parallel, depth), same model/quant as a 1070 control. Find where the card's
   advantage evaporates (somewhere par4→par24 at depth). Suspects, in order: paged-attention decode
   kernel at high-parallel × depth on ROCm/HIP; tiered KV read path; per-step host overhead. Use
   rocprofv3 --kernel-trace (worked great this session) — ablate, don't theorize. This is the last
   piece of the original "long-context tanks earlier than expected" question.
2. Commit the MAD-348 work (after cleanup + review). It's tested and good; it just needs to land.
3. Capture the RX 480 murmur prefill/decode (timed out today).
4. Optional: warm-decode murmur aggregate (steady-state, not cold-burst) for a fair fleet
   deep-context number.

## Method notes that paid off (bank these)
- rocprofv3 --kernel-trace + summary on the 6900 XT (gfx1030) works cleanly; per-kernel durations
  named the (non-)culprit and refuted a wrong theory before I cut a kernel.
- Isolating prefill from decode (n_predict=1 vs cache_prompt reuse) is what made the depth finding
  legible. Always separate the phases.
- "A flat knob sweep is the fingerprint of a fixed cost elsewhere" — the R9700's flat 550-605
  across par AND instances was bandwidth, confirmed by power (135W/300W at 100% use).
- Tests SILENTLY SKIP without `LLAMACPP_TEST_MODELFILE`. Green means nothing without it.
