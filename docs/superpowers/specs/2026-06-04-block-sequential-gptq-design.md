# Block-Sequential GPTQ for Dense ml8 Calibration — Design

**Date:** 2026-06-04 · **Jira:** MAD-264 · **Branch:** `calib-pipeline-speedup`
**Status:** design approved (brainstorm), pending writing-plans.

## Goal

A third `--hessian-mode block-sequential` for the dense ml8 calibration path that is
**exact AND fast**: it preserves cross-layer GPTQ error propagation (the ~0.1 PPL that
static single-pass gives up) while doing ~**2–3 full-forward-equivalents** of compute total
instead of the per-target path's 102 full forwards — i.e. ~30–50× faster than per-target
**and** recovers its quality. Validated first on **Qwen3.5-0.8B dense-hybrid**, PPL-gated
against the per-target reference.

## Strategic context (why this matters)

- **Three modes, three jobs.** `single` (static-Hessian) = the arch-agnostic *fast test bed*
  (default working mode; iterate here). `per-target` (true-sequential, 102 forwards) = the
  slow *exact reference*. `block-sequential` = a *per-arch turbo* that is both exact and fast,
  for high-volume arch families.
- **Dense is now a first-class win target** (kmbandy, 2026-06-02): the "concede dense to UD,
  the win is MoE-shaped" framing is **retired** — "the most popular open-source model right
  now is a 27B dense." Block-sequential's causal walk is also the enabling substrate for the
  two deferred seams (§7) that answer UD's mixed precision on principled, category-level terms.
- **Paging bonus.** The walk touches weights one block at a time in forward order — the access
  pattern that does *not* thrash the NVMe pager (unlike per-target's repeated full-model
  sweeps). So block-sequential is the natural fit for the paging-bound large-model path too,
  not just the 0.8B bed.

## Correct framing (do not relapse)

- The per-target loop is **true-sequential** GPTQ: it writes each quantized weight back so the
  next target's Hessian sees quantized upstream (`calibrate_ml8_paged.py:1860`, `:2071`). It is
  **not** redundant re-forwarding.
- But the per-target *order* is a tier/kind-grouped enumeration **artifact**
  (`find_dense_full_targets`), **not causal**. So block-sequential's bar is **principled causal
  propagation, PPL-gated vs per-target — NOT bit-identity** with that arbitrary order. Because
  block-sequential is *more* causal than the reference, it can match or **beat** it.
- The right gate fits each thing: refactor → **bit-identical**; `run_block` reproduction →
  **fp-noise**; the algorithm change → **PPL within/below noise**.

## Architecture & components (Approach A — self-contained)

Three components; the slow per-target reference path stays behaviorally pristine.

1. **`quantize_one_target(name, layer, H, n_tok, sum_abs, rotation_hook, args, recipe, out_dir,
   manifest)`** — extracted *verbatim* from the current per-target loop body
   (`calibrate_ml8_paged.py:1963–2110`: AWQ rescale → rotation → `batched_gptq_quantize` →
   inverse-rotation/AWQ-absorb → writeback → save blob → manifest). Both the per-target loop
   and the block-sequential walk call it; persistence is **not** duplicated.
   - `recipe` is a new parameter (`group_size`, `n_centroids`) that **defaults to the global
     args** — the cheap door-opener for the §7 per-role seam. Logic stays uniform in this build.
   - **Gate (Tier 0):** the per-target path must produce **bit-identical** blobs before/after
     the extraction.

2. **`block_sequential.py`** (the ~150-line new module) — owns the catcher/replay walk and the
   arch-adapter seam (§Adapter). Calls `quantize_one_target`; does not re-implement persistence.

3. **Driver branch** — `--hessian-mode block-sequential` branches *before* the per-target loop
   into `block_sequential.run(...)`, then **skips** the per-target loop. The embed pass
   (`:2129+`) and FP8 pass (`:2178+`) run **unchanged** afterward (neither propagates;
   order-independent).

   - **Consequence (matched to reference):** FP8 tensors (`ssm_out`/`ffn_down`/`attn_v`) are
     quantized *after* the main loop in **every** mode, so even per-target propagates through
     **bf16** versions of those during its walk. Block-sequential matches this exactly →
     apples-to-apples gate. Whether FP8 should join propagation is a deliberate future question,
     not this build.

## Data flow

**Phase 0 — Catch (one partial forward).** A Catcher pre-hook on `model.model.layers[0]`
records, per calib sample, the block's positional args + **all kwargs** (attention mask,
`position_ids`, rotary `position_embeddings`, `cache_position`, any SSM state slots), then
raises a sentinel to abort the rest of the forward. Run all calib samples up to block 0.
Output `inps = [(args_s, kwargs_s) for s in calib]`. Rotary `position_embeddings` are
model-level/identical across blocks → captured once, reused. Block-0 input uses **bf16 embed**
(embed quantized data-free *after* the walk, matching per-target).

**Phase 1 — Walk.** For block `b` in `0…N-1`:
- **(a) Collect H** — enable Hessian accumulation on block `b`'s ML8 linears (existing
  `FaithfulActHook` + `set_hessian_target`), run `block_b(*args_s, **kwargs_s)` over every
  cached sample, disable. Block `b`'s weights are still **original** here (correct: H is over
  the layer's *inputs*, before its own weight is quantized).
- **(b) Quantize** — for each ML8 target in `b`, in dependency-sub-group order, call
  `quantize_one_target(...)`; writeback leaves the block holding quantized weights.
- **(c) Propagate** — run `block_b(*args_s, **kwargs_s)` **again** with quantized weights,
  capture each sample's **output** hidden_states → becomes block `b+1`'s input cache. Free
  block `b`'s input cache.

Two single-block forwards per block (one for H, one to propagate) = ~2 full-forward-equivalents
(more with intra-block sub-groups, §Adapter — still ~2–3 total). They cannot fuse: quantization
happens between (a) and (c).

**Phase 2 — Tails (unchanged):** embed pass + FP8 pass.

**Memory:** two activation caches only at the b→b+1 hand-off (~330 MB @ 80k, ~1 GB @ 256k) +
**only the current block's** Hessians (vs `single`-mode's 102 simultaneous). Lighter on
Hessian memory than the single-pass mode we already run. Caches may spill to CPU and stream
per-block for larger models.

## The arch-adapter seam

The walk is mostly arch-agnostic; per-arch knowledge concentrates in a thin adapter:

```python
class BlockArchAdapter:
    def iter_blocks(model)             -> list[nn.Module]    # qwen35: model.model.layers
    def ml8_targets(block, block_idx)  -> list[SubGroup]     # ordered dependency groups of (name, linear)
    def run_block(block, args, kwargs) -> (output_hidden, next_kwargs)   # single-block forward
```

**What is reused free from HF transformers (the heavy lifting):** `run_block` is, at core,
**calling the HF module** — `model.model.layers[b](hidden, **kwargs)` runs HF's *own* validated
per-arch forward (delta-net scan, attention, gating). We reimplement **no** architecture math.
When HF ships a new model, `model.model.layers` and `block.__call__` already exist → block-
sequential rides it. Kwargs are free too — the Catcher captures whatever HF passes; `run_block`
replays it. **We reuse HF transformers (PyTorch), NOT llama.cpp** (its per-arch work is the
GGML *inference* graph — wrong framework for a PyTorch calibration forward).

**What we write (thin, declarative):** (1) block location (`model.model.layers`, near-universal
→ a **default adapter** with zero per-arch code); (2) which linears are ML8 targets + their
dependency sub-groups — **ours** (our tier scheme), but **reuses the existing role classifier**
(`find_dense_full_targets`/`classify_role` already enumerate qwen35 ML8 targets); the new bit is
just grouping them; (3) an SSM-state override **only if the §Testing Tier-2 gate demands one**.

**Design:** a default adapter that works for any standard HF decoder model out of the box, plus
a per-arch override **only when the validation gate fails**. The gate is the tripwire that says
whether any per-arch code is needed at all.

**Intra-block granularity (the one real sub-decision) — true-sequential sub-groups (chosen).**
`ml8_targets` returns dependency **sub-groups**, not a flat list. Per block: `[q,k] →
[attn_output]`, `[gate,up]` (`v`,`down` are FP8/done-later). The block is re-forwarded between
groups so `attn_output`'s H sees quantized `q,k`. This is the **GPTQModel blueprint default**
(`true_sequential=True`) and is *strictly more causal* than the non-causal per-target reference,
so it can only help the gate; cost stays ~2–3 forward-equivalents total. The sub-group structure
*is* the per-arch dependency knowledge — declaring it is the adapter's job.

**Discipline:** the adapter's target enumeration is grounded in a **live tensor/module probe**
of the model, not from memory ("every model now is a hybrid — probe before you assume").

## Error handling & resume

Four fail-loud guards:
1. **Adapter equivalence gate fails** → hard abort at *setup*, before any quantization,
   reporting block-kind + max abs diff (the SSM-state tripwire).
2. **Catcher abort** uses the AutoGPTQ sentinel-exception pattern; **fallback** if an HF forward
   swallows it: run the probe forward fully, capture via a plain hook, discard downstream.
3. **`quantize_one_target` fails mid-block** → reuse the existing snapshot-restore
   (`:2043–2046`), but **abort the block** loudly (a half-quantized propagated block would
   poison everything downstream).
4. **Non-finite propagated activations** → after (c), scan the output cache for NaN/Inf; abort
   with the block index (catch divergence at its source, not as a garbage PPL an hour later).

**Resume disabled by design** — block-sequential is minutes; the per-target resume machinery
assumes a contiguous quantized *prefix*, which doesn't map to a block-walk mid-state.
`--resume` + `block-sequential` → `SystemExit`, mirroring the `single`-mode guard at `:1909`.

## Testing & the PPL gate ladder

| Tier | Test | Bar |
|---|---|---|
| 0 | `quantize_one_target` extraction: diff per-target blobs before/after | **bit-identical** |
| 1 | Propagation unit test (toy 2–3 block): block 1's H reflects block 0's **quantized** output | structural (propagation provably occurs) |
| 2 | Adapter `run_block` equivalence, per block kind (attention + SSM): capture block I/O on a reference forward, replay, compare | **fp-noise** (SSM-state tripwire, before any GPU run) |
| 3 | N=1 reduction: block-sequential vs static on a 1-block subset | bit-identical / fp-noise (no propagation to differ at N=1) |
| 4 | **PPL acceptance (full, GPU) — the real number** | **PPL within/below noise vs per-target** |
| 5 | Phase-timing confirms ~2–3 forward-equivalents vs per-target's 102 | ~30–50× holds |

**Tier 4 detail:** full block-sequential calibration on Qwen3.5-0.8B (80k, then 256k), convert
with **`ML8_TIER_OVERRIDE` exported** (the confounding-run footgun), **tight** PPL on full
wiki.test + held-out (not 8-chunk ±1.3), apples-to-apples (498 MB, embed quantized). Two
comparisons: (a) vs **per-target** — match or **beat** (≤ per-target + noise); (b) vs **static
single-pass** — close most/all of the ~0.1 PPL gap.

**Efficiency:** the offset-stability experiment running now (3 percdamp × {static, per-target}
@ 80k) is *pre-computing the per-target reference numbers* Tier 4 gates against — no need to
re-run per-target to grade block-sequential.

## Deferred seams (out of scope for this build; doors held open)

Both are **category-level** (per-role), not UD-style per-tensor learned; both serve the dense
win goal.
1. **Per-role recipe config (bpv trading).** Generalize the existing per-kind `group_size`
   override (`:2006–2008`) into a per-**role** recipe map carrying `group_size` (+ optionally
   `n_centroids`). Mechanism for "4.125 on robust roles / 4.50 on sensitive, param-weighted mean
   ≈ 4.25." Note: ml8 bpv = `index_bits + scale_bits/group_size`; index_bits move only in whole
   bits (codebook size), the fractional part is the scale overhead → finer/coarser `group_size`
   per role is the fine knob; values land on a lattice, not arbitrary decimals; the budget is a
   **param-weighted mean**. (Confirm the on-disk scale dtype before writing the lattice into a
   plan.)
2. **Propagated per-role sensitivity report.** The walk holds the accumulated-error state → it
   is the only place that can emit a deployment-faithful per-role/per-block Y_SNR report (the
   manifest already records per-target `y_snr_db` at `:2106`). This is the measurement that
   would later inform seam 1's allocation.

**This build's only seam work:** the `recipe` default-to-global parameter on
`quantize_one_target`. Parameter exists; per-role logic does not.

## Out of scope / non-goals

- FP8 tensors joining propagation (matched-to-reference bf16 during the walk for now).
- The allocation *logic* for §7 seams (only the parameter hook).
- Dual-GPU / multi-arch beyond the default adapter + qwen35 (the gate tells us when a new arch
  needs an override).
