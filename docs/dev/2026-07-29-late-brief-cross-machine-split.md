# 2026-07-29 late-evening brief — GLM-5.2 cross-machine split

Companion to `docs/dev/2026-07-29-evening-brief.md` (the 2.39x scheduling work).
That brief still stands; nothing here contradicts it.

Read section 0 first, then section 1. Section 1 is the thing most likely to bite
you before coffee.

---

## 0. FIRST THING TOMORROW: the `.wpb` loader

The repack is done, byte-verified, and **completely inert**. 238 GB sitting on
main that nothing can read. This is the agreed first task.

### What exists

`tools/wp-repack` produced, from the 7-shard original:

```
76 shards, 19,456 groups (76 layers x 256 experts), 58,368 members
238,169,358,336 bytes  (= the exact expert total computed independently)
/home/kmbandy/models/GLM-5.2-repack/glm52-experts-{00001..00076}-of-00076.wpb
                                    + matching .wpi.json per shard
                                    + glm52-experts-manifest.json
verify PASS  (compares indexes AND every payload byte against the model)
write 4m57s, verify 3m36s
```

### What is missing

Nothing outside `tools/wp-repack/` and `tests/test-wp-repack.cpp` reads the
format. The pager resolves pages through `(file_idx, file_offset)` on the
original GGUFs (`wp-page-catalog.h:31-32`, consumed by
`WeightPager::ensure_odirect_fd_()` at `wp-pager.cpp:688` via
`file_io_->fd(file_idx)`).

### Why it is a contained change

The sidecar already records exactly the mapping needed. Real entry from
`glm52-experts-00001-of-00076.wpi.json`:

```json
{ "blob_file": "glm52-experts-00001-of-00076.wpb",
  "layer_first": 3, "layer_last": 3, "group_count": 256,
  "groups": [ { "block_idx": 3, "expert_idx": 0, "member_count": 3,
    "members": [
      { "role_mask": 1, "size": 3637248, "offset": 0,
        "catalog_name": "blk.3.ffn_up_exps.weight#expert.0",
        "source_tensor_name": "blk.3.ffn_up_exps.weight",
        "source_file_idx": 1, "source_file_offset": 4439585152 },
      { "role_mask": 2, "size": 3637248, "offset": 3637248,
        "catalog_name": "blk.3.ffn_gate_exps.weight#expert.0", ... },
      ...
```

`catalog_name` **is** the PageCatalog page name. So the loader is: build
`catalog_name -> (blob file, blob offset, size)` from the sidecars at init,
register the 76 `.wpb` files with `file_io_`, and rewrite each expert page's
`(file_idx, file_offset)` to the blob. Everything downstream (O_DIRECT
alignment, the pool, ensure_batch) is unchanged because it only ever sees
`(file_idx, file_offset)`.

### The payoff, and why it is worth doing before more measurement

Within a group the three members are contiguous (`offset` 0, 3637248, ...), so
**one expert becomes ONE contiguous read instead of three scattered ones**. For
blk.3 that is a single 12.09 MB read replacing 3.64 + 3.64 + 4.82 MB. Tonight's
measured effective I/O was 5.51 GB/s against a 6.25 GB/s O_DIRECT ceiling, and
the remaining ~13% is exactly the kind of gap fewer, larger, sequential reads
close.

### Design notes for whoever specs it

- Gate it behind a flag/env so the original-GGUF path stays the default until
  proven. There is precedent for the failure mode: `WP_SIZE_CLASS_SLOTS` has
  been implemented and unit-tested since this morning and has **never run on
  GLM**.
- The manifest carries `content_hash` and `model_files`; refuse to load a blob
  set whose `model_files` do not match the model being opened. A silently
  mismatched blob set is wrong weights, not an error.
- `--verify` exists and works. Re-run it after any change to the writer.
- Blobs live at `/home/kmbandy/models/GLM-5.2-repack/`; that directory has
  `chattr +C` set (see section 8 on the btrfs trap).

---

## 1. NOTHING IS COMMITTED. SIX CHANGESETS ARE LOOSE IN THE WORKING TREE.

`mad-lab-main`, `mad-lab-2026`, and origin are all at **`9475687be`**. Every fix
below is uncommitted on main only. `mad-lab-2026` has none of it.

```
 M common/arg.cpp                               (16 +-)
 M src/llama-graph.cpp                          (14 +-)
 M src/llama-model.cpp                          ( 3 +-)
 M src/llama-pipeline.cpp                       (16 +-)
 M src/llama-pipeline.h                         ( 9 +-)
 M src/models/deepseek2.cpp                     (10 +-)
 M tests/test-pipeline-band.cpp                 (84 +-)
 M tests/test-wp-stage-split.cpp                (161 +-)
 M tools/pipeline/pipeline.cpp                  ( 1 -)
 M tools/wp-stage-split/wp-stage-split-lib.cpp  (221 +-)
 10 files, 383 insertions, 152 deletions
?? docs/dev/2026-07-29-{stage-split-multishard,mtp-head-ownership,arg-registration-completion}-spec.md
```

Also present and **NOT OURS** — do not touch, do not stage:
`ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/*` (the
DSWS kernel spike, kmbandy's uncommitted work).

**First git action tomorrow: commit these by explicit path, then push, then
merge on 2026.** Six logical commits, in dependency order:

1. `wp-stage-split`: multi-shard input support (+ its test)
2. MTP/NextN layer ownership -> head, `output*` duplicated (`llama-pipeline.*`,
   `llama-model.cpp`, `test-pipeline-band.cpp`)
3. `common/arg.cpp`: register the 8 weight-paging/pipeline options for
   `LLAMA_EXAMPLE_COMPLETION`
4. `llama-graph.cpp`: fix the use-after-move in `build_inp_hidden`
5. `tools/pipeline/pipeline.cpp`: remove the null `batch.token` write
6. `src/models/deepseek2.cpp`: decide tail-vs-head from the band, not tensor
   presence

Do **not** `git add -A` or `commit -a` — the aiter tree is dirty.

---

## 2. THE HEADLINE: the pipeline correctness gate ran for the first time, and passes

`llama-pipeline` two-stage vs single-process `llama-server`, GLM-5.2
UD-Q2_K_XL, greedy, seed 0, temp 0, loopback on mad-lab-main:

```
prompt "Once upon a time, there was a", n=32
  ref : " thing called "The Internet". It was a place where people could go to
          talk about things, share things, and learn things. It was a place where people"
  pipe: identical, byte for byte
PASS
```

**32 consecutive argmax decisions agreeing exactly.** That is the load-bearing
result: a structurally wrong partition (skipped layer, double norm, misindexed
positions, mangled hidden) cannot produce it — every such bug tonight diverged
within one or two tokens.

### The honest caveat

Two of three prompts still FAIL:

| prompt | n | result | reference output |
|---|---|---|---|
| "Once upon a time, there was a" | 32 | **PASS** | coherent prose |
| "The meaning of life is" | 16 | FAIL | `:0:\n 1: 'a'\n 2: 'b` |
| "def fibonacci(n):" | 24 | FAIL | ` metrically 0.0.0.0.0.0` |

Both failures have **degenerate reference output** — the model has no confident
continuation. Interpretation: splitting a graph changes op fusion and reduction
order, and the hidden state round-trips through an F32 buffer, so results differ
in the last bits. Invisible when the top-2 logits are far apart; decisive when
near-tied, and greedy decoding then cascades because every later token sees
different context.

**This interpretation is UNMEASURED.** I was about to measure the top-2 logit
gap at the divergence point (via `llama-server`'s `n_probs`) when we stopped.
Until that is done, "numerical tie-break" is a plausible story, not a finding.
Do not write it down as fact.

Also note exact-token equality is arguably the wrong gate for a split graph —
bit-exactness is not achievable in general. The rigorous gate is perplexity
equality (what validated DS4-Flash at identical 1.9007), but `llama-perplexity`
has no pipeline driver, so that is a build task. kmbandy's view: PPL "is just
going to give us a number to tell us what's going on" — so this is not a
priority, but the gate's limitation should be stated wherever the pass is cited.

---

## 3. The split, as decided and executed

`mad-lab-main` = **head, blocks 0-54 + blk.78 (MTP)**.
`mad-lab-2026` = **tail, blocks 55-77**.

Chosen by balancing **measured** drive bandwidth, not layer count:

```
main  6.25 GB/s (O_DIRECT pool)      2026  3.08 GB/s (QD8), 2.1 GB/s (QD1)
2026 target share = 3.08/(6.25+3.08) = 33.0%   ->  band 55-77 gives 32.0%
```

2026's drive is a **WD Black SN750 250 GB** (`WDS250G3X0C-00SJG0`, DRAM-backed --
an earlier revision of this brief called it a DRAM-less SN550, which was wrong on
both counts). Its ~3.1 GB/s rated sequential read matches the 3.08 GB/s measured
here, so the drive is AT SPEC, not underperforming. It is the *same SSD that hosts
the MCP, dashboard and mneme daemon*. Its band must always leave that box real
headroom. Alternatives considered: 53-77 gave a better 34.6% share but left only
21 GB free.

Stage-split output, verified:

```
head [0,54]  1280 of 1809 tensors  164.99 GiB (177.17 GB)  27 blk.78 tensors
tail [55,77]  531 of 1809 tensors   71.94 GiB ( 77.25 GB)   0 blk.78 tensors
1280 + 531 = 1811 selected, 1809 unique -> the ONLY duplicates are
  output.weight and output_norm.weight (deliberate, see section 4)
both stages: correct pipeline.layer_first/last, NO leaked split.* keys
tail transferred to 2026, sha256 420becc5...23b385 identical both ends
```

Model geometry, for reference: `glm-dsa`, 253.9 GB, `block_count` 79 but
`n_layer` resolves to **78** (`nextn_predict_layers=1`). Expert layers are
blocks 3-77 = 75. Experts 238.2 GB, non-expert 14.5 GB, globals 1.19 GB.
Mean 3.134 GB/block, range **3.095-4.178** — mixed quant, so never estimate band
sizes from one block's tensor sizes (I did exactly that from blk.78 and was
wrong).

### Timing, since the estimate was disputed and kmbandy was right

Writing the tail 1m44s, the head 4m57s, rsync of 77 GB at 104 MB/s ~12 min.
Whole split-and-ship cycle under 20 minutes, not the "couple of hours" I said.

---

## 4. MTP: what we learned, and what it still needs

`blk.78` is **not a small head**. It is a full MoE transformer block:

```
expert tensors  3.498 GB  (its own 256 experts)  -> pages like any other layer
dense tensors   257.7 MB                          -> the only resident cost
total           3.756 GB (27 tensors)
```

Critically: **no `blk.78.nextn.embed_tokens` and no
`blk.78.nextn.shared_head_head`** — both declared `TENSOR_NOT_REQUIRED` in
`src/models/glm-dsa.cpp` and absent from the file. So the MTP head is *tied* and
needs the model-level `token_embd` **and** `output`.

That is why MTP goes on the **head**: main already owns `token_embd`, so only
`output.weight` (535 MB) had to be duplicated. Hosting MTP on the tail instead
would have meant duplicating `token_embd` (654 MB) — 119 MB worse — and
`llama-pipeline`'s existing tail->head `PIPE_TOKEN` frame
(`pipe-protocol.h:109`) already flows every decode step, so the hidden state
MTP consumes can ride it with **no new round trip**.

### Still required before MTP works — three items, one of them a landmine

1. **`glm-dsa.cpp` deliberately skips the whole MTP layer.** Loader side is
   nearly free — the tensors are already declared with correct shapes including
   `eh_proj [2*n_embd, n_embd]`:
   ```cpp
   for (int i = 0; i < n_layer_all; ++i) {
       int flags = 0;
       if (i >= n_layer) {
           // skip all tensors in the NextN layers
           flags |= TENSOR_SKIP | TENSOR_NOT_REQUIRED;
       }
   ```
   That is why every `blk.78` tensor logs "unused tensor ... ignoring". My
   earlier claim that MTP "is not implemented for glm-dsa" was reasoned from
   grepping the wrong file (`src/llama-model.cpp` instead of `src/models/`) —
   the conclusion held (no nextn graph exists) but the loader is far cheaper
   than I implied.
2. **`llama-model.cpp:2433` WILL throw the moment you drop `TENSOR_SKIP`.**
   There is a band guard, separate from the ownership predicate, that rejects
   any *pager-catalog* tensor outside the stage band:
   ```
   "pipeline: pager catalog contains out-of-band tensor '%s' (band is [%d, %d])"
   ```
   `blk.78` is outside the head's `[0,54]`. It is harmless today only because
   `TENSOR_SKIP` keeps those tensors out of the catalog. Fix this in the same
   change or the head will refuse to load.
3. The `build_nextn` graph itself (enorm(h) + hnorm(emb(t)) -> `eh_proj` ->
   block 78 -> `shared_head_norm` -> `output`), the `PIPE_TOKEN` hidden-state
   field, and the draft/verify loop. There is a detailed prior wiring plan for
   DeepSeek-V4-Flash in the KG that is a close template.

---

## 5. Five bugs found and fixed tonight — all in code that had never executed

1. **Weight-paging args not registered for `LLAMA_EXAMPLE_COMPLETION`**
   (`common/arg.cpp`). `tools/pipeline/pipeline.cpp:550` parses with
   COMPLETION; all 8 paging/pipeline options were `{SERVER, CLI, PERPLEXITY}`
   only. `add_opt` (`arg.cpp:1211`) never even *creates* an option outside the
   active example, so **`set_env` is not consulted either** —
   `LLAMA_ARG_WEIGHT_PAGING=1` is not a workaround. A 2-stage split of a 254 GB
   model always leaves one stage far bigger than 46 GB of VRAM, so this made the
   feature unreachable. Third instance of this gap (`llama-completion` too).
2. **Use-after-move null deref in `llm_graph_context::build_inp_hidden`**
   (`llama-graph.cpp`). `res->add_input(std::move(inp))` then `inp->hidden`
   twice. Crashed **every** mid-band stage in context creation, 100%. Fixed by
   binding `auto & cur = inp->hidden` before the move — the idiom the adjacent
   `build_inp_pos()` already uses. Sole caller is `src/models/deepseek2.cpp:191`.
3. **Null `batch.token[i]` write in `fill_embd_batch`**
   (`tools/pipeline/pipeline.cpp:191`). `llama_batch_init(n, embd, 1)` allocates
   *either* `embd` *or* `token` (`src/llama-batch.cpp`), never both. Killed the
   tail on its first `FWD_REQ`. One line deleted. Sibling scan of all 23
   `add_input(std::move(...))` sites and both batch-fill loops: clean.
4. **Stage role inferred from tensor presence** (`src/models/deepseek2.cpp`).
   `if (model.output_norm != nullptr)` was the proxy for "am I the tail". **I
   broke this myself** with the MTP ownership change: giving the head `output*`
   flipped it onto the tail branch, so the head applied the final RMSNorm and
   exported a normalized vector. Symptom: fluent-but-different text, diverging
   at token 0. Now
   `!model.pipeline_band_enabled() || il_last == n_layer-1`. Cross-arch scan for
   the same pattern: **none**.
5. **`wp-stage-split` had no multi-shard support** — one `gguf_init_from_file`,
   one `ifstream`. Every band on a 7-shard model selected zero tensors. Mirrored
   `wp-repack`'s enumeration (`wp-repack.cpp:212-238`), and stripped
   `split.no`/`split.count`/`split.tensors.count` from the single-file output so
   a stage never advertises itself as a shard.

---

## 6. Still broken, or worked around — do not assume these work

- **Band resolution ignores the stage GGUF.** `pipeline.cpp` (~596) calls
  `llama_pipeline_resolve_band(params.pipeline_layer_first, ..., n_layer)` —
  CLI values only. The band the loader adopts goes into `mparams`, a different
  struct, never read here. Without `--pipeline-layers` the band silently falls
  back to the full range, the tail computes its peer as `{s.last+1, n_layer-1}`
  = `{78,77}`, and the handshake dies with `layer band [78, 77] is empty`
  — three steps from the cause. **The comment at that call claims it falls back
  to GGUF metadata. It does not.** Spec is written and staged at
  `~/.claude/jobs/87d16c2e/tmp/band-resolution-spec.md`, ready to hand off.
  Workaround in use: always pass `--pipeline-layers` explicitly.
- **Related, same root cause:** `band_from_cli` is computed *before* load
  (`pipeline.cpp:561`), so a non-head stage relying on GGUF metadata never gets
  `--no-warmup` forced. There is already a warning in the code admitting this
  ("this should have failed by design").
- **`--fit` is hostile to pipeline stages.** On by default, it does a full trial
  `llama_init_from_model` to measure memory. For a 177 GB paged stage that
  doubles load time, and it crashed *before logging started*, which is why the
  first failure showed 0 bytes of output. Harness passes `--fit off`.
- **`llama-pipeline` has no standalone mode** — it requires `--pipeline-peer` or
  `--pipeline-listen`. It cannot be the reference arm.
- **`tools/pipeline/loopback-test.sh` cannot gate GLM.** It re-splits the model
  into a temp dir (another 254 GB) and runs CPU-only with paging off (254 GB
  into 15 GB of RAM). The working harness is
  `~/.claude/jobs/87d16c2e/tmp/glm_gate.sh` (also at `/tmp/wpx/` on main),
  which reuses the stage files and runs on GPU with paging on.

---

## 7. "Believed complete, actually not" — the pattern, and the inventory

The recurring shape: **code that compiles, has unit tests, and has never been
executed against the target model.** The common cause is that the only
band-capable model on the fleet is 254 GB, so nothing could be exercised
cheaply — and the gate that would have caught all of it was written but never
run. The moment it ran, it found five bugs in ninety minutes.

| capability | assumed | actual |
|---|---|---|
| `wp-repack` | done | writes + verifies; **no read path**, inert |
| Phase 2 pipeline | done | never executed; 4 bugs, every mid-band stage crashed |
| `loopback-test.sh` | the gate | written, never run, cannot run on GLM |
| paging args on pipeline | assumed | not registered; feature unreachable |
| band from GGUF metadata | comment says so | never read; still broken |
| non-head warmup suppression | assumed | warning admits it does not fire |
| MTP on glm-dsa | "preserved but unused" | declared then `TENSOR_SKIP`'d, no graph |
| `WP_SIZE_CLASS_SLOTS` / pin floors | shipped this morning | **never run on GLM** |
| Vulkan H2D overlap | parity | runs, overlaps nothing, counters blind |

Standing offer, no GPU and no writes: a pass over the weight-pager and pipeline
surface marking every capability **measured working** / **runs but unproven** /
**written, never executed**, with the evidence for each.

---

## 8. Prefetch is dead for the current config (from earlier tonight)

Recorded in the KG as decision `99874443`. Summary, because it closes a line of
work:

- **The "1,621 near-1.0 cross-layer chains" number is refuted.** It counted
  links with >=5 observations, where 5/5 scores 1.00 by luck, with no baseline.
  Threshold sweep with LIFT: 1621 links >=0.90 at min_obs 5 -> **10 at min_obs
  100, of which ZERO have lift >=2** -> 0 at 200. Prompt B agrees. Artifact.
- **Real signal does exist**: held-out predictor (fit on first half of steps,
  scored on second) gets **51.1% / 34.3% recall** at M = n_used = 8 vs 3.1%
  chance and 22.9-27.1% for frequency-only.
- **It still loses, on bandwidth.** `total/baseline = M/n_used + (1-recall)`;
  neutrality needs 100% precision, so every imperfect prefetcher *increases*
  bytes read and only idle drive time pays. Measured 5.51 of 6.25 GB/s =>
  **HEADROOM 13%**. Gate `M/n_used <= recall + 0.13` passes only at M = 1-3
  pages of 8 (~2% net). **A perfect predictor caps at 1.13x.**
- Prefetch needs drive utilization <= ~51%, i.e. hit rate >= ~48%, i.e. a
  ~46 GB pool arena — the whole fleet. **Prefetch is a consequence of pool
  capacity, not a substitute for it.** Revisit only after the pool grows.

Related, and cheap: the 2.39x run used **2500 uniform slots x 6,684,672 B**, and
that slot size comes from `attn_k_b.weight`, a **non-expert** tensor. Expert
pages are 3.6-5.4 MB. Roughly a third of a 15.56 GB arena is padding, and
`WP_SIZE_CLASS_SLOTS` (this morning's pin-floor work) would recover it — for one
env var, never yet tried on GLM.

---

## 9. Corrections to my own claims tonight

Recording these because several were confidently wrong.

1. **"MTP is not implemented for glm-dsa"** — reasoned from grepping
   `src/llama-model.cpp` when per-arch code lives in `src/models/`. The loader
   declares everything; it is `TENSOR_SKIP`'d.
2. **Device ordering.** I trusted `rocm-smi` (`GPU[0]` = 16 GB 6900 XT,
   `GPU[1]` = 32 GB R9700) and swapped the harness accordingly. llama.cpp's
   `--list-devices` is the **reverse**: `ROCm0` = R9700 32624 MiB, `ROCm1` =
   RX 6900 XT 16368 MiB. Result: I put the 177 GB stage on the 16 GB card, where
   ~15.7 GB of resident dense left no room, and the pool failed with
   `allocating 3825.00 MiB on device 1: cudaMalloc failed`. **Always read
   `--list-devices`.**
3. **Page-size constants from `blk.78`.** UD-Q2_K_XL is a mixed quant; blk.78 is
   also the unused MTP layer. Empirical `io_bytes/page_ins` = 4.075 MB is the
   trustworthy average.
4. **"A couple of hours" for the split and transfer.** It was ~20 minutes.
   kmbandy called it correctly.
5. **`--log-disable` on the reference arm** suppressed the generated text
   itself; the reference exited rc=0 with 0 bytes. The harness's own comments
   warn about this.
6. **I used `llama-completion` as the reference** after Terra's arg fix made it
   available, against a standing instruction to use **`llama-server` only**.
   Corrected. (The one useful by-product: `llama-server` produced byte-identical
   output, which is what established the baseline is trustworthy.)
7. **I briefly suspected the Q2 quant** on the basis of short factual prompts
   producing non-sequiturs. That is normal base-model behaviour at temp 0 with
   no chat template; "Once upon a time" was perfectly coherent. Unsloth's UD
   quants are well-tested — dropped.
8. **I skipped the repack** when it was explicitly asked for, on my own judgment
   that it was pointless without a loader. Then when pushed I over-corrected
   into a 34 GB bounded run, then a 238 GB full run. The write ordering should
   have been **loader first, then repack** — and I should have said that
   *before* spending 238 GB of writes, not after.

---

## 10. Method rules earned tonight

1. **Run the gate before believing a feature.** Every one of tonight's five bugs
   was in code with tests that had never been executed end to end.
2. **Read `--list-devices`, never infer device index from `rocm-smi`.**
3. **A crash with zero log output is probably a pre-flight probe** (`--fit`)
   dying before logging starts, or buffered output lost on SIGSEGV. `--fit off`
   plus `stdbuf -o0 -e0` before concluding "it dies instantly".
4. **gdb over elimination.** Two backtraces localized two null derefs in minutes
   after ~20 minutes of narrowing by hypothesis got nowhere.
5. **Frame #0 with no library frames beneath it means the fault is in inlined
   code in that function** — that is what pointed at `fill_embd_batch` rather
   than `llama_decode`.
6. **N consecutive exact tokens is a structural proof.** 32 matching argmax
   decisions cannot come from a wrong partition; it is stronger evidence than
   any single-token comparison.
7. **A link count needs an observation floor AND a lift baseline.** Without
   both, small-sample noise reads as a discovery.
8. **When a feature spends a resource that is already saturated, compute the
   budget before measuring the predictor.**
9. **Do not narrow an explicit instruction on your own judgment.** If the cost
   looks wrong, say so and let the decision be made — before spending the cost.

---

## 11. Open items, ranked

1. **The `.wpb` loader** (section 0). Agreed first task.
2. **Commit and propagate the six changesets** (section 1). Do this before any
   further edits or the tree gets confusing.
3. **Band resolution from GGUF metadata** (section 6). Spec written and staged;
   small; removes a landmine whose error message points nowhere near the cause.
4. **Cross-machine Gate B.** Never run. Needs main's head stage regenerated
   (deleted tonight, 4m57s) and 2026's tail (already there, sha256-verified).
   2026's tail is the least-tested config in the stack: two 8 GB cards on
   *different backends* (CUDA 1070 + Vulkan RX 480) in one process, and the
   Vulkan pager's counters are known blind.
5. **Measure the tie-break hypothesis** (section 2) so the gate's limitation is
   a finding rather than a story.
6. **MTP enablement**: `TENSOR_SKIP`, the `llama-model.cpp:2433` band guard, the
   `build_nextn` graph, `PIPE_TOKEN` hidden field, draft/verify loop.
7. `WP_SIZE_CLASS_SLOTS=1` on GLM — one env var, ~1.5x the pages in the same
   arena, never tried.
8. Microbatching / >=2 tokens in flight. **Nothing about the split pays without
   it**: stages serialize at 0.655 + 0.565 s ~= today's 1.39 s. Pipelined it is
   max(0.655, 0.565) => ~1.53 t/s, a 2.1x.

---

## 12. Housekeeping / state

- **Board claims still held**: `31850c89` (gpu:R9700), `83b8aa94`
  (gpu:RX6900XT), both on mad-lab-main, 3h TTL from ~21:00. **Release them** if
  the morning does not start with GPU work.
- **main**: 98 GB free. Originals intact (7 shards). Stage files **deleted**
  (254 GB reclaimed). Repack present, 222 GiB / 238.2 GB, verified.
- **2026**: 36 GB free. `glm52-tail-55-77.gguf` present, 77.25 GB,
  sha256 `420becc5...23b385`.
- **Drive writes tonight on main's SN850X**: ~254 GB (stages) + 34 GB (an
  orphan from an interrupted repack call, since deleted) + 238 GB (repack)
  ~= **526 GB**, roughly 0.09% of the drive's ~600 TBW. Not a concern per se,
  but it repeats per model, and Kimi K2.7 Code is 318 GB.
- `/tmp` on main is **tmpfs, ~1.7 GB**. Harnesses must write to `/var/tmp`.
- `GLM-5.2-repack/` and `GLM-5.2-stages/` both have `chattr +C`. main's `/home`
  is btrfs with `compress=zstd:1`, and **O_DIRECT cannot be served from a
  compressed extent**. Tonight's stage file happened to be incompressible
  (apparent 177,170,121,312 vs on-disk 177,170,124,800, and O_DIRECT read at
  2.3 vs 2.6 GB/s against a `+C` original), so it dodged the trap — but any new
  model directory needs `+C` set *before* writing.
- Throwaway harnesses, on main at `/tmp/wpx/` and locally in
  `~/.claude/jobs/87d16c2e/tmp/`: `glm_gate.sh` (the working gate),
  `tail_bt.sh`, `coherence.sh`, `tiebreak.sh` (written, not run),
  `verify_stage.py`, `prefetch_gate.py`, `lru_sim.py`, `band_bytes.py`,
  `mtp_tensors.py`, `verify_bytes.py` (written, not run — superseded by
  `wp-repack --verify`).

---

## 13. Kimi K2.7 Code — the fallback target, unverified

kmbandy raised it as the alternative if Kimi K3 does not land: coding-specialized,
1.1T, 8 experts/token, **UD-Q2 = 318 GB**, KLD 0.3241, PPL 2.4131 vs 1.8419 at
full fidelity.

The decisive unknown, cheaply checkable from HF metadata without downloading:
**what `general.architecture` does the GGUF declare?** Band capability today is
exactly the archs whose graph is `llama_model_deepseek2::graph` — `deepseek2`,
`deepseek2ocr`, `glm_dsa`, `mistral4` — because that is the only graph calling
`build_inp_hidden()`. Kimi K2 is DeepSeek-V3-architecture-derived, so if K2.7
declares `deepseek2` it is band-capable **with zero new arch work**, unlike K3
(only `LLM_ARCH_KIMI_LINEAR` exists in the tree, and it is a different model).
Verify before committing to a 318 GB download.

Arithmetic worth having in advance: bytes/token depends on **active** experts,
not total. 8 active x ~12.5 MB/expert x ~61 layers ~= **6.1 GB/token**, i.e.
about the same as GLM-5.2's measured 6.26 GB/token despite being 1.5x larger on
disk. Decode speed should land in the same ballpark. Disk is the real constraint:
318 GB original + a 213 GB main band means retiring GLM-5.2 from main, and
2026's ~105 GB bandwidth-optimal band barely fits its 226 GB system SSD — I
would deliberately undersize 2026's band there.
